import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
import json
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

def inference_and_save(image_path, prompt, model, processor, output_file):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Result: {output_text[0]}\n")

    print(f"Saved results to: {output_file}")
    return output_text[0]

if __name__ == "__main__":
    # Load model
    model_path = "/home/ubuntu/ocr/DOTS_OCR/dots.ocr/weights/DotsOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.gradient_checkpointing_enable()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    image_path = "demo_image1.jpg"

    # Process with different prompts and save results
    for i, (prompt_mode, prompt) in enumerate(dict_promptmode_to_prompt.items()):
        output_file = f"ocr_result_{i+1}_{prompt_mode}.txt"
        print(f"\nProcessing: {prompt_mode}")
        result = inference_and_save(image_path, prompt, model, processor, output_file)