### Image recognition → Video idea generation → ComfyUI video generation.
#### ✔ All running locally.

## Features

- 🖼️ Batch processing of multiple images
- 🤖 Automatic image description using Moondream2
- ✍️ Creative prompt generation using Qwen
- 🎬 Integration with ComfyUI workflow API
- 📝 Comprehensive logging
- 🧠 Smart memory management for GPU resources
- 🎨 Designed for easy modification and experimentation
- 
## Example
#### Input Image
![sphoto_2026-01-28_14-30-13](https://github.com/user-attachments/assets/8dfd6e00-4433-4462-bdaa-57d27102bca4)
#### Generated description
- A white-winged angel, wearing a flowing white dress, is kneeling on the dark blue water, its hands resting gently on the calm surface. The angel appears to be mourning or in deep thought, with a somber expression. The sky above is a deep blue, dotted with dark grey clouds, and sunlight pierces through the clouds, casting a warm glow on the angel and the water. Several white feathers are scattered around the angel and on the water's surface, adding to the eerie, dreamlike atmosphere of the scene.
#### Generated video idea
- An angel in a white dress with white wings is kneeling upon a dark blue water surface, lightly touching it with lowered hands. Light breaks through heavy, shadowed clouds overhead, bathing the figure and the water in a soft, warm glow. White feathers drift and rest across the scene, reinforcing a quiet, otherworldly atmosphere. The mood feels somber and ethereal, and the camera stays completely fixed.
#### Final video result
https://github.com/user-attachments/assets/1f3bf24b-a5a7-4969-ab57-002bf36c85db

## Requirements

- Python 3.8+
- CUDA-capable GPU
- [ComfyUI-portable](https://github.com/Comfy-Org/ComfyUI) for video generation. Other versions not tested

- Required models:
  - image-to-text model like [Moondream2](https://huggingface.co/vikhyatk/moondream2) for generating image descriptions
  - a light [Qwen2.5-3B-Instruct-Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) LLM model is enough to generate video ideas
- The models can be downloaded manually. In this case you can set paths to the models in the script.

## Installation

- Create a new folder, e.g. *comfy-video-prompter*
- Inside the folder, run the following commands:
```bash
python -m venv .
```
```
.\Scripts\activate
```
- Run the folloiwng command. This will download and install about 2~4 GB of packages, the biggest ones are for PyTorch and CUDA.
```
pip install -r requirements.txt
```
  
## Configuration

### Required AI Models:
On first run, the project will download these models:
- Moondream2 (image-to-text recognition, ~4GB in size)
- Qwen 2.5 3B Instruct (LLM for generating video ideas, ~2GB in size)

Or you can use already downloaded models if you have them.
- Open main.py and adjust:
```
LLM_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct-GGUF"
```
to the full path of the GGUF model on your drive, for example:
```
LLM_MODEL_PATH = r"C:\AI\text-generation-webui\user_data\models\qwen2.5-3b-instruct-q4_k_m-qwen.gguf"
```

### ComfyUI Workflow Node IDs

Must be changed according to your workflow. Reference the provided workflow IDs.

## Usage

1. Place your images in the `input_images` folder of this project. They will be copied to Comfy's input folder in the process.

2. Ensure ComfyUI is running, usually on http://127.0.0.1:8188.

3. Run the script:
```bash
python main.py
```

4. Monitor progress:
   - Console output shows real-time progress
   - Logs saved to `logs/` directory
   - Video generation happens in ComfyUI and can be monitored in "View Job History" and "Console"
