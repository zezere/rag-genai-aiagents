# Exercises for Diogo's course on RAG, AI Agents and Generative AI with Python and OpenAI 2025

This repository contains my implementations of exercises and code examples from Diogo's course on RAG, AI Agents, and Generative AI. The course materials were originally provided as Jupyter Notebook files, but I have restructured and replicated the code in a traditional Python project format. This structure is ideal for those who prefer working with Python scripts instead of Notebooks.

## Repository Structure

- **`src/`**: Contains the Python scripts for the exercises. Each script is well-documented with comments to guide you through the code and its functionality.
- **`data/`**: Includes pre-downloaded datasets required for the exercises. If you wish to download fresh copies or learn how to fetch the data programmatically, the relevant code is included in the scripts but commented out.

## How to Use This Repository

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zezere/rag-genai-aiagents.git
   ```
2. **Explore the code**:
   Navigate to the **`src/`** folder, where you'll find Python scripts, each corresponding to an exercise in the course.
3. **The order of execution**:
   1. `basics_of_retrieval_systems.py` (Section 3)
   2. `basics_of_generation_models.py` (Section 5)
   3. `introduction_to_rag.py` (Section 7)
   4. `openai_api.py` (Section 9)
   5. `image_to_excel_genai.py` (Capstone project, section 10)
   6. `pdf_to_image_genai.py` (Capstone project, section 10)

## Requirements
- See `requirements.txt` for the full list of dependencies
- M1/M2 Mac users should follow the special setup instructions below

## Important Notes for M1/M2 Mac Users

The RAG implementation (`introduction_to_rag.py`) has been modified from the original course material to ensure compatibility with Apple Silicon (M1/M2) Macs. Here's what you need to know:

1. **Environment Setup**:
   - Create a Python 3.9 environment (PyTorch has known issues with newer Python versions on M1/M2):
     ```bash
     conda create --name your-environment-name python=3.9
     conda activate your-environment-name
     ```
   - Install PyTorch:
     ```bash
     pip3 install pytorch torchvision torchaudio
     ```
   - Install remaining packages:
     ```bash
     pip install -r requirements.txt
     ```

2. **Implementation Changes**:
   - Instead of using the raw `transformers` library, we use `sentence-transformers`.
   - This change provides better stability and memory management on M1/M2 Macs.

