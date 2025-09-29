import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import fitz  # PyMuPDF
import gc
import time
from pathlib import Path
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse

class BatchPDFProcessor:
    def __init__(self, model_path="/home/ubuntu/ocr/DOTS_OCR/dots.ocr/weights/DotsOCR"):
        """Initialize the batch PDF processor with optimized settings"""
        print("Loading DOTS OCR model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.gradient_checkpointing_enable()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("Model loaded successfully!")

    def pdf_to_images(self, pdf_path, temp_dir, max_pages=None, dpi=150):
        """Convert PDF pages to images with memory-efficient approach"""
        print(f"Converting PDF to images (DPI: {dpi})...")

        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)

        if max_pages:
            total_pages = min(total_pages, max_pages)

        print(f"Processing {total_pages} pages from {pdf_path}")

        image_paths = []
        matrix = fitz.Matrix(dpi/72, dpi/72)  # Convert DPI to zoom factor

        for page_num in range(total_pages):
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=matrix)

            image_path = os.path.join(temp_dir, f"page_{page_num+1:04d}.png")
            pix.save(image_path)
            image_paths.append((page_num + 1, image_path))

            # Clean up pixmap to save memory
            pix = None

            if (page_num + 1) % 10 == 0:
                print(f"Converted {page_num + 1}/{total_pages} pages")

        pdf_doc.close()
        return image_paths

    def process_single_page(self, image_path, page_num):
        """Process a single page with memory cleanup"""

        prompt = """Please extract all the text from this document in a clean, readable format.
Provide ONLY the text content in markdown format with proper headers and formatting.
Do NOT provide JSON output, bounding boxes, or any structured data.
Just give me the clean text as it appears in the document."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process the request
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate response with memory-efficient settings
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=24000,  # Back to original capacity with flash-attn
                do_sample=False,
                temperature=0.1
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Clean up the output - remove any JSON artifacts
        cleaned_text = self.clean_output_text(output_text)

        # Clean up GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()

        return page_num, cleaned_text

    def clean_output_text(self, text):
        """Clean up output text to remove JSON formatting"""
        text = text.strip()

        # If it looks like JSON, try to extract text content
        if text.startswith('[{') or text.startswith('{'):
            import json
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    clean_text = ""
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            clean_text += item['text'] + "\n\n"
                    return clean_text.strip()
            except:
                pass

        return text

    def process_large_pdf(self, pdf_path, output_path=None, max_pages=None, dpi=150):
        """Process large PDF with batch processing and memory optimization"""

        pdf_name = Path(pdf_path).stem
        if output_path is None:
            output_path = f"{pdf_name}_extracted.md"

        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")

            # Convert PDF to images
            image_paths = self.pdf_to_images(pdf_path, temp_dir, max_pages, dpi)
            total_pages = len(image_paths)

            print(f"\nStarting OCR processing for {total_pages} pages...")
            start_time = time.time()

            # Process pages sequentially to avoid memory issues
            results = []

            for i, (page_num, image_path) in enumerate(image_paths):
                print(f"Processing page {page_num}/{total_pages}...", end=" ")

                page_start = time.time()
                page_num, page_text = self.process_single_page(image_path, page_num)
                page_time = time.time() - page_start

                results.append((page_num, page_text))

                # Memory usage monitoring
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                    print(f"Done ({page_time:.1f}s, GPU: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached)")
                else:
                    print(f"Done ({page_time:.1f}s)")

                # Progress update every 10 pages
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (total_pages - i - 1)
                    print(f"Progress: {i+1}/{total_pages} pages, {elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining")

            # Sort results by page number and combine
            results.sort(key=lambda x: x[0])

            print(f"\nCombining results into {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# {pdf_name}\n\n")
                f.write(f"*Extracted from {total_pages} pages using DOTS OCR*\n\n")
                f.write("---\n\n")

                for page_num, page_text in results:
                    f.write(f"## Page {page_num}\n\n")
                    f.write(page_text)
                    f.write(f"\n\n---\n\n")

            total_time = time.time() - start_time
            avg_time_per_page = total_time / total_pages

            print(f"\n✅ Processing completed!")
            print(f"📄 Total pages: {total_pages}")
            print(f"⏱️  Total time: {total_time:.1f}s")
            print(f"📈 Average per page: {avg_time_per_page:.1f}s")
            print(f"💾 Output saved to: {output_path}")

            return output_path

def main():
    parser = argparse.ArgumentParser(description="Process large PDFs with DOTS OCR")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Output markdown file path")
    parser.add_argument("-p", "--pages", type=int, help="Maximum number of pages to process")
    parser.add_argument("-d", "--dpi", type=int, default=150, help="DPI for PDF to image conversion (default: 150)")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file {args.pdf_path} not found")
        return

    print("🚀 DOTS OCR Batch PDF Processor")
    print("================================")

    processor = BatchPDFProcessor()
    processor.process_large_pdf(
        pdf_path=args.pdf_path,
        output_path=args.output,
        max_pages=args.pages,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()