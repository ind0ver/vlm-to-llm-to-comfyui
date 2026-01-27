import json
import os
import glob
import torch
import requests
import datetime
import gc
import shutil
import random
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from PIL import Image
from transformers import AutoModelForCausalLM
from llama_cpp import Llama
import instruct_templates

# ==================== CONFIGURATION ====================

INPUT_IMAGE_DIR = "./input_images"
LOG_DIR = "./logs"
WORKFLOW_JSON_PATH = "./MyImageToVideoWorkflowAPI.json"
COMFYUI_PATH = ".../ComfyUI-portable/ComfyUI" # full path to ComfyUI-portable/ComfyUI
COMFYUI_API_URL = "http://127.0.0.1:8188"

# Models
LLM_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct-GGUF" # or you can use a full path to your local GGUF model file
llm_model_types = {"llama": "llama", "qwen": "qwen"}
llm_model_type = llm_model_types["qwen"]
VLM_MODEL_NAME = "vikhyatk/moondream2" # or you can use locally downloaded image-to-text model

# Replace these node IDs with your actual workflow node IDs
NODE_LOAD_IMAGE = "160" # node Load Image - the input image for the video
NODE_POSITIVE_PROMPT = "93" # node Positive Prompt - for the video description
NODE_VIDEO_COMBINE = "169"  # node Video Combine by VHS - for the saved video file name

# ==================== TYPE DEFINITIONS ====================

@dataclass
class ComfyTask:
    input_image_path: str = ""
    input_image_filename: str = ""
    image_description: str = ""
    prompt_for_video: str = ""

# ==================== LOGGING UTILITIES ====================

run_start_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

def log(message: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_file = os.path.join("logs", f"log-{run_start_time}.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} {message}\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

# ==================== STARTUP UTILITIES ====================

def check_dependencies() -> bool:
    # Verify directories exist
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"Image directory not found: {INPUT_IMAGE_DIR}")
        return False
    
    if not os.path.exists(WORKFLOW_JSON_PATH):
        print(f"Workflow JSON not found: {WORKFLOW_JSON_PATH}")
        return False
    
    if not os.path.exists(LLM_MODEL_PATH):
        print(f"LLM model not found: {LLM_MODEL_PATH}")
        return False
    
    if not os.path.exists(COMFYUI_PATH):
        print(f"ComfyUI directory not found: {COMFYUI_PATH}")
        return False

    return True


def ping_comfyui() -> bool:
    try:
        r = requests.get(f"{COMFYUI_API_URL}/system_stats", timeout=5)
        return r.ok
    except requests.RequestException:
        return False


def load_workflow_base(workflow_path: str) -> dict:
    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        log(f"Failed to load workflow JSON: {e}")
        raise RuntimeError("Workflow JSON load failed") from e
    
    required_nodes = {
    NODE_LOAD_IMAGE: ("inputs", "image"),
    NODE_POSITIVE_PROMPT: ("inputs", "text"),
    NODE_VIDEO_COMBINE: ("inputs", "filename_prefix"),
    }
    
    for node_id, (section, key) in required_nodes.items():
        if node_id not in workflow:
            raise KeyError(f"Workflow missing node '{node_id}'")
    if section not in workflow[node_id]:
        raise KeyError(f"Node '{node_id}' missing section '{section}'")
    if key not in workflow[node_id][section]:
        raise KeyError(f"Node '{node_id}' missing field '{section}.{key}'")

    return workflow


def prepare_tasks() -> List[ComfyTask]:
    comfy_tasks: List[ComfyTask] = []

    image_files: List[str] = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(INPUT_IMAGE_DIR, ext)))
    
    for image_file in image_files:
        image_filename = os.path.basename(image_file)
        comfy_tasks.append(ComfyTask(input_image_path=image_file, input_image_filename=image_filename))

    return comfy_tasks
    
# ==================== IMAGE-TO-TEXT MODEL ====================

def load_i2t_model() -> Any:
    try:
        model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_NAME,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        )
        return model
    except Exception as e:
        log(f"VLM load failed: {e}")
        raise RuntimeError("Failed to load image-to-text model") from e


def unload_i2t_model(model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def describe_single_image(model: Any, image_path: str):
    try:
        image = Image.open(image_path)
        # Optionally set sampling settings
        # settings = {"temperature": 0.5, "max_tokens": 180, "top_p": 0.3}
        return model.caption(
            image, 
            length="normal", # "short" | "normal"
            # settings=settings
        )
    except Exception as e:
        log(f"Image description failed [{image_path}]: {e}")
        return ""

# ==================== TEXT GENERATION MODEL ====================

def load_llm_model() -> Any:
    try:
        return Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=4096,
        n_threads=12,
        n_gpu_layers=28,
        )
    except Exception as e:
        log(f"LLM load failed: {e}")
        raise RuntimeError("Failed to load LLM") from e


def unload_llm_model(model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        from llama_cpp import llama_backend_free
        llama_backend_free()
    except ImportError:
        print("Outdated llama-cpp-python version, please update")
    except Exception as e:
        print(f"Error freeing backend: {e}")
    finally:
        print("LLM model unloaded.")


def generate_prompt_for_video(model: Any, description: str) -> str:
    params = {
        'max_tokens': 256,
        'temperature': 0.8,
        'top_p': 0.92,
        'top_k': 50,
        'stop': ["\nInstruct:", "Output:", "\n", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>"],
        'echo': False,
        'repeat_penalty': 1.1,
        'seed': random.randint(1, 2147483647)
    }

    try:
        if llm_model_type == llm_model_types["llama"]:
            prompt = instruct_templates.build_llama_prompt(description)
        elif llm_model_type == llm_model_types["qwen"]:
            prompt = instruct_templates.build_qwen_prompt(description)
        else:
            print(f"Unknown LLM model type: {llm_model_type}.")
            return ""
        
        response = model(prompt, **params)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        log(f"Prompt generation failed: {e}")
        return ""

# ==================== COMFYUI INTEGRATION ====================

def send_to_comfyui(workflow_data: Dict) -> bool:    
    try:
        r = requests.post(
        f"{COMFYUI_API_URL}/prompt",
        json={"prompt": workflow_data},
        timeout=30,
        )
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        log(f"ComfyUI request failed: {e}")
        return False

# ==================== MAIN PROCESS ====================

def main() -> None:
    if not check_dependencies():
        print("Missing dependencies. Check log for details.")
        return

    # ComfyUI must be running
    if not ping_comfyui():
        print(f"ComfyUI is not running at {COMFYUI_API_URL}.")
        return

    # List of tasks to be sent to ComfyUI = the amount of images in the input folder
    comfy_tasks = prepare_tasks()
    if not comfy_tasks:
        print(f"No images found in the input folder: {INPUT_IMAGE_DIR}.")
        return
    
    # Load and validate JSON workflow (in ComfyUI: File -> Export (API))
    workflow_base = load_workflow_base(WORKFLOW_JSON_PATH)

    log(f"{len(comfy_tasks)} images found.")
    print(f"{len(comfy_tasks)} images found.")

    # Load image-to-text model
    print("Loading image-to-text model...")
    model_i2t = load_i2t_model()
    
    # Generate descriptions for each image
    for comfy_task in comfy_tasks:
        print("Reading the image...")
        comfy_task.image_description = describe_single_image(model_i2t, comfy_task.input_image_path)
        if not comfy_task.image_description:
            log(f"Could not generate description for {comfy_task.input_image_filename}!")
            print(f"Could not generate description for {comfy_task.input_image_filename}!")
            continue

        print("\n" + "DESCRIPTION GENERATED " + "=" * 60)
        print(comfy_task.image_description)
        print("\n" + "=" * 60)
        
    unload_i2t_model(model_i2t)
    
    # Generate animation prompts
    print("Loading text-generation model...")
    model_llm = load_llm_model()
    
    for i, comfy_task in enumerate(comfy_tasks):
        comfy_task.prompt_for_video = generate_prompt_for_video(model_llm, comfy_task.image_description)
        if not comfy_task.prompt_for_video:
            log(f"Could not generate description for {comfy_task.input_image_filename}!")
            print(f"Could not generate description for {comfy_task.input_image_filename}!")
            continue
        
        print("\n" + f" VIDEO PROMPT GENERATED ({i + 1}/{len(comfy_tasks)})" + "=" * 60)
        print(comfy_task.prompt_for_video)
        print("\n" + "=" * 60)

    unload_llm_model(model_llm)

    for comfy_task in comfy_tasks:
        log(f"""
            IMAGE:
            {comfy_task.input_image_filename}
            I2T_MODEL:
            {VLM_MODEL_NAME}
            DESCRIPTION:
            {comfy_task.image_description}
            LLM_MODEL:
            {os.path.basename(LLM_MODEL_PATH)}
            PROMPT_FOR_VIDEO:
            {comfy_task.prompt_for_video}
            \n""")

    # Send prompts to ComfyUI    
    print(f"Sending tasks to ComfyUI...")
    sent_tasks_counter = 0

    for i, comfy_task in enumerate(comfy_tasks):
        # Copy image to ComfyUI input folder
        input_folder = os.path.join(COMFYUI_PATH, "input")
        try:
            shutil.copy2(comfy_task.input_image_path, input_folder)
        except Exception as e:
            log(f"Failed to copy image {comfy_task.input_image_path} to ComfyUI input folder: {e}")
            continue
        
        # Create workflow copy for each task
        current_workflow = json.loads(json.dumps(workflow_base))

        # Change values in workflow before sending to Comfy
        current_workflow[NODE_LOAD_IMAGE]["inputs"]["image"] = comfy_task.input_image_filename # node Load Image - the input image for the video
        current_workflow[NODE_POSITIVE_PROMPT]["inputs"]["text"] = comfy_task.prompt_for_video # node Positive Prompt - for the video description
        current_workflow[NODE_VIDEO_COMBINE]["inputs"]["filename_prefix"] = f"{comfy_task.input_image_filename}_{run_start_time}" # node Video Combine by VHS
        
        # Send to ComfyUI
        task_sent_ok = send_to_comfyui(workflow_data=current_workflow)
        if not task_sent_ok:
            print(f"Failed to send task for {comfy_task.input_image_filename}.")
        else:
            sent_tasks_counter += 1

        print(f"Task sent to ComfyUI ({i + 1}/{len(comfy_tasks)})\nImage: {comfy_task.input_image_filename}\nPrompt: {comfy_task.prompt_for_video}")
        print("\n" + "=" * 60)
        
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"{sent_tasks_counter}/{len(comfy_tasks)} tasks sent to ComfyUI!")
    print("Monitor progress in ComfyUI interface.")


if __name__ == "__main__":
    main()
