import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import base64  # as in previous exercises, we convert images to base64 strings for OpenAI
from pdf2image import (
    convert_from_path,
)  # needs installing for this exercise (pip install pdf2image)
import json
import numpy as np
import faiss


def pdf_to_images(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF to images
    images = convert_from_path(pdf_path)
    image_paths = []

    # Save images to output folder
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        image.save(image_path, "JPEG")
        image_paths.append(image_path)

    return image_paths


def query_embeddings(query, index, metadata, k=5):
    # Generate embeddings for the query
    query_embedding = (
        client.embeddings.create(
            input=[query],
            model="text-embedding-3-large",
        )
        .data[0]
        .embedding
    )
    print(f"The query embedding is {query_embedding[:2]} ... etc")
    query_vector = np.array(query_embedding).reshape(1, -1)
    print(f"The query vector is {query_vector[:2][:2]} ... etc")

    # Search faiss index
    distances, indices = index.search(query_vector, min(k, len(metadata)))
    print(f"The distances are {distances[:2][:2]} ... etc")
    print(f"The indices are {indices[:2][:2]} ... etc")

    # Store indices and distances
    stored_indices = indices[0].tolist()
    stored_distances = distances[0].tolist()
    print(f"The stored indices are {stored_indices[:2]} ... etc")
    print(f"The stored distances are {stored_distances[:2]} ... etc")

    # Print metadata content
    print("Metadata content is:\n--------------------------------")
    for i, dist in zip(stored_indices, stored_distances):
        if 0 <= i < len(metadata):
            print(
                f"Distance: {dist:.4f},\nMetadata:\n{metadata[i]['recipe_info'][:50]}"
            )
    print("--------------------------------")

    # Return results
    results = [
        (metadata[i]["recipe_info"], dist)
        for i, dist in zip(stored_indices, stored_distances)
        if 0 <= i < len(metadata)
    ]

    return results


if __name__ == "__main__":

    print("\n\nRAG WITH OPEN AI\n")

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
    PATH_TO_PDFS = os.path.join(project_root, "data", "pdfs")
    PATH_TO_OUTPUT = os.path.join(project_root, "data", "output")

    print("Setup complete.")

    # ====================================================================
    # Step 2. Transform PDFs to images
    #
    # We will create a function that takes a path to a PDF generates
    # images into a directory of the speciafied path.
    # ====================================================================

    print("\n\nTRANSFORM PDFs TO IMAGES\n")

    # Function to convert PDF to images is defined above
    # Here we test it with one pdf: "Things mother used to make.pdf"
    pdf_path = os.path.join(PATH_TO_PDFS, "Things mother used to make.pdf")
    image_paths = pdf_to_images(pdf_path, PATH_TO_OUTPUT)
    print(
        f"Extracted {len(image_paths)} images from PDF and saved them in {PATH_TO_OUTPUT}"
    )

    # ====================================================================
    # Step 3. Read single image with GPT
    # ====================================================================

    # print("\n\nREAD SINGLE IMAGE WITH GPT\n")

    # # Read and encode one image
    # # image_path = image_paths[22]
    # image_path = os.path.join(PATH_TO_OUTPUT, "page_23.jpg")
    # with open(image_path, "rb") as image_file:
    #     image_data = base64.b64encode(image_file.read()).decode("utf-8")

    # system_prompt = """
    # Please analyze the content of this image and extract any related recipe information.
    # """

    # print(f"Reading image: {image_path}")

    # response = client.chat.completions.create(
    #     model=OPENAI_MODEL,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": [
    #                 "This is an image from a page of a recipe book.",
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{image_data}",
    #                         "detail": "low",
    #                     },
    #                 },
    #             ],
    #         },
    #     ],
    # )

    # gpt_response = response.choices[0].message.content
    # print(f"GPT response: {gpt_response}")

    # ====================================================================
    # Step 4. Enhance system prompt to return structured response
    # ====================================================================

    # print("\n\nENHANCE SYSTEM PROMPT TO RETURN STRUCTURED RESPONSE\n")

    system_prompt2 = """
    Please analyze the content of this image and extract any related recipe
    information into structure components. Specifically, extract the recipe
    title, list of ingredients, step by step instructions, cuisine type,
    dish type, any relevant tags or metadata. The output must be formatted
    in a way suited for embedding in a Retrieval Augmented Generation (RAG)
    system.
    """

    # response = client.chat.completions.create(
    #     model=OPENAI_MODEL,
    #     messages=[
    #         {"role": "system", "content": system_prompt2},
    #         {
    #             "role": "user",
    #             "content": [
    #                 "This is an image from a page of a recipe book.",
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{image_data}",
    #                         "detail": "low",
    #                     },
    #                 },
    #             ],
    #         },
    #     ],
    #     temperature=0,
    # )

    # gpt_response2 = response.choices[0].message.content
    # print(f"GPT response: {gpt_response2}")

    # ====================================================================
    # Step 5. Read all images with GPT
    # ====================================================================

    print("\n\nREAD ALL IMAGES WITH GPT\n")

    extracted_recipes = []
    for image_path in image_paths[21:29]:  # limiting this otherwise it takes very long
        print(f"Processing image: {image_path}")
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt2},
                {
                    "role": "user",
                    "content": [
                        "This is an image from a page of a recipe book.",
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            temperature=0,
        )

        # Extract the content and store it
        gpt_response = response.choices[0].message.content  # Get the response content
        extracted_recipes.append(
            {"image_path": image_path, "recipe_info": gpt_response}
        )  # Store the path and extracted info
        print(
            f"Extracted information for {image_path}\nResponse preview: {gpt_response[:100]} ... etc\n"
        )  # Print the extracted info

    # ====================================================================
    # Step 6. Filter out non-recipe content
    # based on key recipe-related terms
    # ====================================================================

    print("\n\nFILTER OUT NON-RECIPE CONTENT\n")

    filtered_recipes = []
    for recipe in extracted_recipes:
        if any(
            keyword in recipe["recipe_info"].lower()
            for keyword in ["ingredients", "instructions", "recipe title"]
        ):
            filtered_recipes.append(recipe)
        else:
            print(f"Skipping recipe: {recipe['image_path']}")

    print(f"Filtered {len(filtered_recipes)} recipes out of {len(extracted_recipes)}")

    # ====================================================================
    # Step 7. Save the filtered recipes to a JSON file
    # ====================================================================

    print("\n\nSAVE THE FILTERED RECIPES TO A JSON FILE\n")

    output_file = os.path.join(PATH_TO_OUTPUT, "recipe_info.json")
    with open(output_file, "w") as f:
        json.dump(filtered_recipes, f, indent=4)

    print(f"Saved {len(filtered_recipes)} recipes to {output_file}")

    # ====================================================================
    # Step 8. Embed the recipes using OpenAI's embedding API
    # ====================================================================

    print("\n\nEMBED THE RECIPES USING OPENAI'S EMBEDDING API\n")

    with open(os.path.join(PATH_TO_OUTPUT, "recipe_info.json"), "r") as f:
        recipes = json.load(f)

    recipe_texts = [recipe["recipe_info"] for recipe in filtered_recipes]

    embedding_response = client.embeddings.create(
        input=recipe_texts,  # list of recipe texts as input
        model="text-embedding-3-large",
    )

    embeddings = [data.embedding for data in embedding_response.data]

    print(f"Generated embeddings for {len(embeddings)} recipes")

    # Convert embeddings to numpy array
    # This is making it effective when it comes to feeding it to our model
    embeddings_matrix = np.array(embeddings)

    print(f"Generated embeddings matrix: {embeddings_matrix[:2][:2]} ... etc")
    print(f"Each embedding is of size {len(embeddings[0])}")

    # ====================================================================
    # Step 9. Build FAISS index and Metadata integration
    # ====================================================================

    print("\n\nBUILD FAISS INDEX AND METADATA INTEGRATION\n")

    print(f"Embedding matrix shape: {embeddings_matrix.shape}")

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    # Save the index to a file
    faiss.write_index(
        index, os.path.join(PATH_TO_OUTPUT, "filtered_recipe_index.index")
    )
    print(
        f"Index saved to {os.path.join(PATH_TO_OUTPUT, 'filtered_recipe_index.index')}"
    )

    # Save the metadata to a file
    # This allows us to look at the actual outcomes
    metadata = [
        {
            "recipe_info": recipe["recipe_info"],
            "image_path": recipe["image_path"],
        }
        for recipe in filtered_recipes
    ]

    with open(os.path.join(PATH_TO_OUTPUT, "recipe_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # ====================================================================
    # Step 10. Query based on embeddings
    #
    # Function to query embeddings will be defined above
    # and then tested here.
    # ====================================================================

    print("\n\nQUERY BASED ON EMBEDDINGS\n")

    query = "how to make bread"
    print(f"Testing with query: {query}\n")
    results = query_embeddings(query, index, metadata)
    print(f"\nA couple of first results are:\n{results[:2][:50]} ... etc")

    # ====================================================================
    # Step 11. Combine results
    # ====================================================================

    print("\n\nCOMBINE RESULTS\n")

    combined_content = "\n\n".join([result[0] for result in results])
    print(f"Combined content:\n{combined_content[:500]} ... etc")

    # ====================================================================
    # Step 12. Construct generative model
    #
    # Constructing a generative model is a three-(sub)step process:
    #   Substep I: Define system prompt
    #   Substep II: Define function to retrieve from API
    #   Substep III: Get the results
    # ====================================================================

    print("\n\nCONSTRUCT GENERATIVE MODEL\n")

    # Substep I: Define system prompt
    system_prompt3 = """
    You are highly experienced and expert chef specialized in providing cooking advice.
    Your main task is to provide information precise and accurate on the combined content.
    You answer diretly to the query using only information from the provided {combined_content}.
    If you don't know the answer, just say that you don't know.
    Your goal is to help the user and answer the {query}
    """

    # Substep II: Define function to retrieve from API
    def generate_response(query, combined_content, system_prompt):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
                {"role": "assistant", "content": combined_content},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    # Substep III: Get the results
    query = "how to make bread"
    print(f"Testing with query: {query}\n")
    response = generate_response(query, combined_content, system_prompt3)
    print(f"Response:\n{response}\n")
