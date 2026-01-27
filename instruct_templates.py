# System message and few-shot examples can be modified to produce better or custom video ideas

system_msg = """
You are an expert AI video prompt engineer specialized in WAN 2.2 video generation.
Your task is to generate exactly ONE detailed video prompt suitable for text-to-video or image-to-video generation.

Rules:
- The output must consist of 3-4 short sentences.
- Describe the scene first, then the action, then the mood (optionally), then the camera behavior.
- Use active, continuous actions (e.g. breathing, blinking, turning, flowing).
- Describe sequential actions separated by periods.
- Be precise with visual details (colors, materials, lighting).
- Avoid metaphors, abstract language, or vague wording.
- The motion must be subtle, realistic, and fit in a 6-second video.
- Do not repeat the same action multiple times.
- Assume a fixed or gently drifting camera.

Return only the final video prompt, nothing else.
"""

# Few-shot examples
examples = [
    {
        "user": "Create one animation idea for this scene: A knight in polished steel armor stands in a dark, fog-filled forest, holding a longsword upright as moonlight filters through tall trees.",
        "assistant": "A knight in polished steel armor is standing still in a dark, fog-filled forest, holding a longsword upright with both hands. Thin mist is slowly drifting between the tree trunks while the knight subtly shifts his weight and his chest rises with calm breathing. Cold moonlight reflects softly off the metal surfaces. The atmosphere feels tense and quiet. The camera remains fixed at chest height."
    },
    {
        "user": "Create one animation idea for this scene: An old fisherman sits on a wooden dock at sunrise, repairing a fishing net beside calm water with distant birds.",
        "assistant": "An old fisherman with tanned skin is sitting on a weathered wooden dock at sunrise, holding a fishing net across his knees. His hands are slowly pulling and tightening the rope while the net gently moves with the breeze. The water below reflects soft orange light and small ripples move outward. The mood is calm and early-morning quiet. The camera is static at a medium distance."
    },
    {
        "user": "Create one animation idea for this scene: A black and white cat peers from behind a half-open door inside a dimly lit room with sunlight streaming in.",
        "assistant": "A black and white cat with green eyes is peeking from behind a half-open wooden door inside a dim room. Its tail is slowly swaying while its ears twitch slightly as dust particles drift through a beam of sunlight. The cat blinks once and subtly leans forward. The mood is cautious and curious. The camera stays still at floor level."
    },
    {
        "user": "Create one animation idea for this scene: A woman in a red dress stands on a balcony overlooking a city at night with glowing lights.",
        "assistant": "A woman with dark hair and a red dress is standing on a balcony overlooking a city at night. Her hair and dress fabric are gently moving in the wind while distant city lights flicker softly. She slowly turns her head to the side and exhales. The atmosphere feels quiet and contemplative. The camera is fixed behind her at shoulder height."
    }
]

# ========================== SCRIPTS ==========================

def build_qwen_prompt(image_description, system_msg=system_msg, examples=examples) -> str:    
    # Build the prompt step by step
    prompt_parts = []
    
    # System message
    prompt_parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
    
    # Add examples
    for example in examples:
        prompt_parts.append(f"<|im_start|>user\n{example['user']}<|im_end|>")
        prompt_parts.append(f"<|im_start|>assistant\n{example['assistant']}<|im_end|>")
    
    # Add the current request
    user_request = f"Create one animation idea for this detailed scene: {image_description}"
    prompt_parts.append(f"<|im_start|>user\n{user_request}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)


def build_llama_prompt(image_description, system_msg=system_msg, examples=system_msg):
    # Build the prompt step by step
    prompt_parts = ["<|begin_of_text|>"]
    
    # System message
    prompt_parts.append("<|start_header_id|>system<|end_header_id|>")
    prompt_parts.append(system_msg)
    prompt_parts.append("<|eot_id|>")
    
    # Add examples
    for example in examples:
        prompt_parts.append("<|start_header_id|>user<|end_header_id|>")
        prompt_parts.append(example['user'])
        prompt_parts.append("<|eot_id|>")
        prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")
        prompt_parts.append(example['assistant'])
        prompt_parts.append("<|eot_id|>")
    
    # Add the current request
    user_request = f"Create one animation idea for this detailed scene: {image_description}"
    prompt_parts.append("<|start_header_id|>user<|end_header_id|>")
    prompt_parts.append(user_request)
    prompt_parts.append("<|eot_id|>")
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")
    
    return "".join(prompt_parts)
