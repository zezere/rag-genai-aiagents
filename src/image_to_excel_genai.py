import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import base64

if __name__ == "__main__":

    print("\n\nCAPSTONE PROJECT: READING IMAGES AND SAVING TO CSVs WITH OPEN AI\n")

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
    # Step 2. Define system prompt
    # ====================================================================

    print("\n\nDEFINING SYSTEM PROMPT\n")

    system_prompt = """
    Convert image of restaurant menu to .csv format following provided
    template and instructions.
    Each row in the .csv file should represent a single menu item.
    Each column in the .csv file should represent a single attribute of a menu item.
    Result should be a valid .csv file, do not include any other text or formatting.
    Do not enclose the result in quotes.
    List of atrribute names, their descriptions, accepted values and examples
    provided below:
    ---
    Attribute name: CategoryTitle
    Description: Name of category of the menu item in original language
    Accepted values: String, enclosed with double quotes, max 256 characters
    Examples: "Bebidas", "Os Classicos", "Sobremesas"
    ---
    Attribute name: CategoryTitleEn
    Description: Name of category of the menu item in English (optional)
    Accepted values: String, enclosed with double quotes, max 256 characters
    Examples: "Beverages", "Soups", "Desserts"
    ---
    Attribute name: SubcategoryTitle
    Description: Name of subcategory of the menu item in original language (optional)
    Accepted values: String, enclosed with double quotes, max 256 characters
    Examples: "Mochis", "Vinhos"
    ---
    Attribute name: SubcategoryTitleEn
    Description: Name of subcategory of the menu item in English (optional)
    Accepted values: String, enclosed with double quotes, max 256 characters
    Examples: "Mochis", "Wines"
    ---
    Attribute name: ItemName
    Description: Name of the menu item in original language
    Accepted values: String, enclosed with double quotes, max 256 characters
    Examples: "Coca-Cola", "Suco de Laranja"
    ---
    Attribute name: ItemNameEn
    Description: Name of the menu item in English (optional)
    Accepted values: String, enclosed with double quotes, max 256 characters
    Examples: "Coca-Cola", "Orange juice"
    ---
    Attribute name: ItemPrice
    Description: Price of the menu item, without currency symbol, with dot as decimal separator
    Accepted values: Numeric
    Examples: 1.50, 2.55, 3.99
    ---
    Attribute name: Calories
    Description: Caloric content of each menu item (optional)
    Accepted values: Numeric
    Examples: 150, 786
    ---
    Attribute name: PortionSize
    Description: Portion size of each menu item (optional)
    Accepted values: String, enclosed with double quotes, max 256 characters
    Examples: "500ml"
    ---
    Attribute name: Availability
    Description: Current availability of the menu item (optional)
    Accepted values: Numeric, 1 for yes, 0 for no
    Examples: 1, 0
    ---
    Attribute name: ItemDescription
    Description: Detailed description of the menu item in original language (optional)
    Accepted values: String, enclosed with double quotes, max 500 characters
    Examples: "Galinha, Vegetais, coco e aroma decaril"
    ---
    Attribute name: ItemDescriptionEn
    Description: Detailed description of the menu item in English (optional)
    Accepted values: String, enclosed with double quotes, max 500 characters
    Examples: "Chicken, Vegetables & touch of coconut and curry"
    ---
    
    Additional notes:
    - Ensure all data entered follows the specified format to maintain data integrity.
    - Review all data for accuracy and consistency before submitting the result.
    - If you cannot find an attribute for a menu item, leave it blank.

    Example of a valid .csv file:
    CategoryTitle,CategoryTitleEn,SubcategoryTitle,SubcategoryTitleEn,ItemName,ItemNameEn,ItemPrice,Calories,PortionSize,Availability,ItemDescription,ItemDescriptionEn
    "SOPAS","SOUPS","","","Sopa Won Ton","Won Ton Soup",3.95,150,"500ml",1,"Galinha, Vegetais, coco e aroma decaril","Chicken, Vegetables & touch of coconut and curry"
    "SOPAS","SOUPS","","","Sopa Vegetariana","Vegetarian Soup",3.95,150,"500ml",1,"Galinha, Vegetais, coco e aroma decaril","Chicken, Vegetables & touch of coconut and curry"
    """

    print(f"System prompt is: {system_prompt}")

    # ====================================================================
    # Step 3. Open image files in binary mode and encode them in base64
    # ====================================================================

    print("\n\nENCODING IMAGES\n")

    def encode_image(image_path_and_name):
        with open(image_path_and_name, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    dimsum_menu_image_file_names_and_paths = sorted(
        [
            f
            for f in os.listdir(PATH_TO_DIMSUM)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    print(dimsum_menu_image_file_names_and_paths)
    regatta_menu_image_file_names_and_paths = sorted(
        [
            f
            for f in os.listdir(PATH_TO_REGATTA)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    print(regatta_menu_image_file_names_and_paths)

    image_data = []
    for filename in dimsum_menu_image_file_names_and_paths:
        image_data.append(
            {
                "filename": filename,
                "image_data": encode_image(os.path.join(PATH_TO_DIMSUM, filename)),
            }
        )
    for filename in regatta_menu_image_file_names_and_paths:
        image_data.append(
            {
                "filename": filename,
                "image_data": encode_image(os.path.join(PATH_TO_REGATTA, filename)),
            }
        )
    print("Images encoded.")

    # ====================================================================
    # Step 4. Call the API
    # ====================================================================

    print("\n\nCALLING THE API\n")

    for one_image in image_data:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Convert this menu image to structured .csv format",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{one_image['image_data']}"
                            },
                        },
                    ],
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        print(f"Generated CSV content:\n{content}\n")

        # create folder if it doesn't exist
        os.makedirs(PATH_TO_OUTPUT, exist_ok=True)
        with open(f"{PATH_TO_OUTPUT}/{one_image['filename']}.csv", "w") as f:
            f.write(content)
        print(f"CSV file saved to {PATH_TO_OUTPUT}/{one_image['filename']}.csv\n")
