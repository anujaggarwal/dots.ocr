#!/usr/bin/env python3
"""
Simple script to process large PDFs using DOTS OCR
Usage: python process_large_pdf.py your_file.pdf
"""

import sys
import os
from batch_pdf_processor import BatchPDFProcessor

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_large_pdf.py <pdf_file> [max_pages]")
        print("Example: python process_large_pdf.py document.pdf 50")
        sys.exit(1)

    pdf_file = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.exists(pdf_file):
        print(f"Error: File {pdf_file} not found")
        sys.exit(1)

    print(f"Processing PDF: {pdf_file}")
    if max_pages:
        print(f"Maximum pages: {max_pages}")

    # Initialize processor
    processor = BatchPDFProcessor()

    # Process the PDF
    output_file = processor.process_large_pdf(
        pdf_path=pdf_file,
        max_pages=max_pages
    )

    print(f"\n✅ Success! Extracted text saved to: {output_file}")

if __name__ == "__main__":
    main()