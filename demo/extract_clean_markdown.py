import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz  # PyMuPDF

def extract_clean_markdown(image_path, model, processor, output_file):
    """Extract content in clean markdown format from image/PDF"""

    # Simple prompt for clean text extraction without JSON
    simple_prompt = """Extract all the text from this document in clean, readable format.
Use markdown formatting for headers, tables, and emphasis where appropriate.
Do not include any JSON, bounding boxes, or layout information.
Just provide the clean text content in markdown format."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": simple_prompt}
            ]
        }
    ]

    # Process the request
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Clean up the output - remove any JSON artifacts
    if output_text.startswith('[{') or output_text.startswith('{'):
        # Try to extract just text from JSON if it's still in JSON format
        import json
        try:
            data = json.loads(output_text)
            if isinstance(data, list):
                clean_text = ""
                for item in data:
                    if 'text' in item:
                        clean_text += item['text'] + "\n\n"
                output_text = clean_text
        except:
            pass

    # Save to markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text.strip())

    print(f"Clean markdown extracted to: {output_file}")
    return output_text

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

    # Process the demo image
    print("Extracting clean markdown from demo_image1.jpg...")
    extract_clean_markdown("demo_image1.jpg", model, processor, "clean_output_markdown.md")

    # Process the specified PDF
    pdf_file = "Annexure_22_pages_166-166.pdf"
    if os.path.exists(pdf_file):
        print(f"Converting {pdf_file} to image and extracting clean markdown...")

        # Convert PDF to image using PyMuPDF
        pdf_doc = fitz.open(pdf_file)
        page = pdf_doc[0]  # Get first page
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        image_path = "pdf_page_clean.png"
        pix.save(image_path)
        pdf_doc.close()

        print(f"PDF converted to {image_path}")

        # Extract clean markdown from the converted image
        extract_clean_markdown(image_path, model, processor, "clean_pdf_markdown.md")
    else:
        print(f"PDF {pdf_file} not found")