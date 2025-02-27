# This file corresponds to Diogo's Section 9: OpenAI API
import os
from dotenv import load_dotenv
from openai import OpenAI
import base64  # needed for Step 5
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":

    # ====================================================================
    # Step 1. Setup
    #
    # In the Notebook, Diogo connects to Google Drive and gets his API Key
    # via Google colab. We have files locally. As for API key, you should
    # use your own and store it in your .env file - NEVER EVER COMMIT IT
    # TO GITHUB! Then use dotenv to load it into your script.
    # ====================================================================

    print("\n\nSETUP\n")

    # Get your OpenAI API key from .env file
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not _OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Define our model
    MODEL = "gpt-4o"

    # Initialize OpenAI client
    client = OpenAI(api_key=_OPENAI_API_KEY)

    print(f"OpenAI client initialized with model: {MODEL}")

    # ====================================================================
    # Step 2. Generating text with OpenAI
    #
    # Here we will define the system prompt in which we tell OpenAI to
    # adopt a persona of Kendrick Lamar / Taylor Swift.
    # ====================================================================

    # print("\n\nGENERATING TEXT WITH OPENAI\n")

    # system_prompt = "You are Kendrick Lamar"
    # user_prompt = "Write a diss song about Drake with 2 verses and a chorus."

    # # Generate text
    # response = client.chat.completions.create(
    #     model=MODEL,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    # )

    # print(f"System prompt: {system_prompt}")
    # print(f"User prompt: {user_prompt}")
    # print(f"\nGenerated text:\n\n{response.choices[0].message.content}")

    # ====================================================================
    # Step 3. Text Generation with Parameters
    #
    # Here we influence creativity and randomness of the generated text by
    # adjusting parameters like temperature and top_p.
    # ====================================================================

    # print("\n\nTEXT GENERATION WITH PARAMETERS\n")

    # # Generate text
    # response = client.chat.completions.create(
    #     model=MODEL,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     temperature=1.2,
    #     top_p=1.0,
    #     presence_penalty=0.0,
    #     frequency_penalty=0.0,
    # )

    # print(f"System prompt: {system_prompt}")
    # print(f"User prompt: {user_prompt}")
    # print(f"\nGenerated text:\n\n{response.choices[0].message.content}")

    # ====================================================================
    # Step 4. Interacting with Images
    #
    # We will use OpenAI API to analyze an image.
    # ====================================================================

    # print("\n\nINTERACTING WITH IMAGES\n")

    # # URL of the image
    # url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # # Our prompt will consist of two parts:
    # # - question as string and
    # # - image passed via URL.
    # user_content = [
    #     {
    #         "type": "text",
    #         "text": "Describe the image",
    #     },
    #     {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
    # ]
    # # Use chat completions as usually, only the content is now more complex
    # response = client.chat.completions.create(
    #     model=MODEL, messages=[{"role": "user", "content": user_content}]
    # )
    # print(f"\nResponse:\n\n{response.choices[0].message.content}")

    # ====================================================================
    # Step 5. Using Base64 Encoded Images
    #
    # Here we provide image not via URL, but directly,
    # encoding it in Base64.
    # ====================================================================

    print("\n\nUSING BASE64 ENCODED IMAGES\n")

    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent
    file_path = os.path.join(project_root, "data", "images", "Thumbnail python FV1.jpg")

    with open(file_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Print few first characters of the image variable
    print(f"Image base64: {image_base64[:10]}...")

    # Now we can include this image variable directly in our request
    # without needing a publicly accessible URL.
    user_content2 = [
        {
            "type": "text",
            "text": "This is the image for my thumbnail for my Python for Data Analysis course. Be brutal, mean and provide sarcastic suggestions",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "high",
            },
        },
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_content2}],
    )

    print(f"\nResponse:\n\n{response.choices[0].message.content}")
