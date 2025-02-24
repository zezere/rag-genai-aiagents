# This file corresponds to Diogo's Section 9: OpenAI API
import os
from dotenv import load_dotenv
from openai import OpenAI

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

    print("\n\nGENERATING TEXT WITH OPENAI\n")

    system_prompt = "You are Kendrick Lamar"
    user_prompt = "Write a diss song about Drake with 2 verses and a chorus."

    # Generate text
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    print(f"System prompt: {system_prompt}")
    print(f"User prompt: {user_prompt}")
    print(f"\nGenerated text:\n\n{response.choices[0].message.content}")

    # ====================================================================
    # Step 3. Text Generation with Parameters
    #
    # Here we influence creativity and randomness of the generated text by
    # adjusting parameters like temperature and top_p.
    # ====================================================================

    print("\n\nTEXT GENERATION WITH PARAMETERS\n")

    # Generate text
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.2,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    print(f"System prompt: {system_prompt}")
    print(f"User prompt: {user_prompt}")
    print(f"\nGenerated text:\n\n{response.choices[0].message.content}")
