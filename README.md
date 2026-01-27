# vlm-to-llm-to-comfyui
**Image recognition** → **Video idea generation** → **ComfyUI video generation via API** — all running locally on your PC.

## Overview

This tool processes images through three stages:
1. **Image Description** - Uses a Vision-Language Model (VLM) to describe the image
2. **Prompt Generation** - Uses an LLM to create video animation prompts
3. **Video Generation** - Sends prompts to ComfyUI for rendering

## Workflow

```
Input Images
    ↓
[VLM Model] → Image Descriptions
    ↓
[LLM Model] → Video Prompts
    ↓
[ComfyUI API] → Video Generation
```

## Features

- 🖼️ Batch processing of multiple images
- 🤖 Automatic image description using Moondream2
- ✍️ Creative prompt generation using Qwen
- 🎬 Integration with ComfyUI workflow API
- 📝 Comprehensive logging
- 🧠 Smart memory management for GPU resources

## Requirements

- Python 3.8+
- CUDA-capable GPU
- ComfyUI installed and running

- Required models:
  - image-to-text model [Moondream2](https://huggingface.co/vikhyatk/moondream2) for generating image descriptions
  - a light [Qwen2.5-3B-Instruct-Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) LLM model to generate video ideas
The models can be downloaded manually. In this case you can set paths to the models in the script.

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd image-to-video-pipeline
```
or download the .zip archive.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
   - Moondream2 (automatically downloaded via Transformers, ~4GB in size)
   - Qwen 2.5 3B Instruct (can be automatically downloaded via Transformers, ~2GB in size)
  
## Configuration

Open main.py and adjust:
```
COMFYUI_PATH = ".../ComfyUI-portable/ComfyUI" # full path to your ComfyUI-portable/ComfyUI
COMFYUI_API_URL = "http://127.0.0.1:8188"
LLM_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct-GGUF" # or you can use a full path to your local GGUF model file
```

### ComfyUI Workflow Node IDs

Must be changed according to your workflow. Reference the provided workflow IDs.

## Usage

1. Place your images in the `input_images` folder

2. Ensure ComfyUI is running:
```bash
# Start ComfyUI on http://127.0.0.1:8188
```

3. Run the script:
```bash
python main.py
```

4. Monitor progress:
   - Console output shows real-time progress
   - Logs saved to `logs/` directory
   - Video generation happens in ComfyUI interface

## Notes

- All models can be local or remote (HuggingFace)

- Designed for easy modification and experimentation

## Acknowledgments

- [Moondream2](https://huggingface.co/vikhyatk/moondream2) for image recognition
- [Qwen](https://huggingface.co/Qwen) for text generation
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) for video generation

## License
MIT — do whatever you want, attribution appreciated.
