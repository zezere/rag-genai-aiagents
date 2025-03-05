import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# This project needs packages PyMuPDF and Pillow
# Install them with: pip install PyMuPDF Pillow
import fitz  # module from PyMuPDF for PDF handling
from PIL import Image  # module from Pillow for image processing


if __name__ == "__main__":

    print("\n\nCAPSTONE PROJECT: CONVERTING PDF TO IMAGES WITH GENAI\n")

    # ====================================================================
    # Step 1. Set up the environment and main variables
    # ====================================================================

    print("Starting setup...")

    load_dotenv()
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not _OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = OpenAI(api_key=_OPENAI_API_KEY)

    OPENAI_MODEL = "gpt-4o"

    project_root = Path(__file__).resolve().parent.parent
    PATH_TO_DIMSUM = os.path.join(project_root, "data", "dimsum")
    PATH_TO_REGATTA = os.path.join(project_root, "data", "regatta")
    PATH_TO_OUTPUT = os.path.join(project_root, "data", "output")

    print("Setup complete.")

    # ====================================================================
    # Step 2. Find all PDF files
    # ====================================================================

    print("\n\nFINDING ALL PDF FILES\n")

    pdf_files = []
    for file in os.listdir(PATH_TO_DIMSUM):
        if file.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(PATH_TO_DIMSUM, file))
    for file in os.listdir(PATH_TO_REGATTA):
        if file.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(PATH_TO_REGATTA, file))
    print(f"Found {len(pdf_files)} PDF files:\n{pdf_files}\n")

    # ====================================================================
    # Step 3. Opening all PDF files
    # Going through each page of each PDF file and storing it
    # ====================================================================

    print("\n\nOPENING ALL PDF FILES\n")

    for pdf_file in pdf_files:
        print(f"Opening {pdf_file}...")
        pdf_document = fitz.open(pdf_file)
        print(f"Found {len(pdf_document)} pages in {pdf_file}.")

        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            # Save image directly to output folder
            image_path = os.path.join(
                PATH_TO_OUTPUT,
                f"{os.path.splitext(os.path.basename(pdf_file))[0]}_{page_number+1}.jpg",
            )
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(image_path)
            print(f"Saved image {image_path}")
