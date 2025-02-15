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
   Navigate to the **`src/`** folder, where you'll find Python scripts - for now one script for exercises in Section 3 of the course, and another script for Section 5.
3. **Run the code**:
   ```bash
   python src/basics_of_retrieval_systems.py
   python src/basics_of_generation_models.py
   python src/introduction_to_rag.py
   ```

## Important Notes for M1/M2 Mac Users

The RAG implementation (`introduction_to_rag.py`) has been modified from the original course material to ensure compatibility with Apple Silicon (M1/M2) Macs. Here's what you need to know:

1. **Environment Setup**:
   - Create a Python 3.8 environment (PyTorch has known issues with newer Python versions on M1/M2):
     ```bash
     conda create --name your-environment-name python=3.8
     conda activate your-environment-name
     ```
   - Install PyTorch using conda (more stable than pip for M1/M2):
     ```bash
     conda install pytorch::pytorch torchvision torchaudio -c pytorch
     ```

2. **Implementation Changes**:
   - Instead of using the raw `transformers` library, we use `sentence-transformers`:
     ```bash
     pip install sentence-transformers
     ```
   - This change provides better stability and memory management on M1/M2 Macs while achieving the same learning objectives.

3. **Why These Changes?**:
   - The original implementation caused segmentation faults on M1/M2 Macs due to memory management issues
   - `sentence-transformers` provides a higher-level, more stable API that's better optimized for embedding tasks
   - These changes maintain the same educational value while ensuring code runs reliably across different platforms

## Requirements
- See `requirements.txt` for the full list of dependencies
- M1/M2 Mac users should follow the special setup instructions above
