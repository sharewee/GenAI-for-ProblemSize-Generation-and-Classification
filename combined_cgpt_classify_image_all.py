import os
import base64
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load all the keys from the .env file 
load_dotenv()

PROMPT = """
You will view an image telling a short story about a child experiencing a social problem. 
Identify the main problem in the story and classify it into one of three categories based on its size.

Classify the problem as one of the following categories: 
   Problem Size Guide:
    disaster: Posing serious risk to personal health or safety or lost of lives of close friends or family members, or suffer from large financial loss, or require significant help from
        others and long time to recover
    bummer: Disappointing, medium size problems that can't be quickly fixed, may needs time and effort or help from others to solve it over time, not serious, this category is between glitch and disaster
        examples of bummer are Group disagreement, missing homework, misunderstanding with a friend, parent or teacher.
    glitch: Minor annoyance that will pass with time or quickly fixed.

Return only one word — “disaster”, “bummer”, or “glitch” — in lowercase.
Do not include any explanation or extra text/symbols such as quotation marks.
"""

def encode_image(image_path: str) -> str:
    """Encode image as base64 string for OpenAI API"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def predict_problem_size(client: OpenAI, image_path: str) -> str:
    """Use GPT-4o to classify the size of the problem from an image"""
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Here is the image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }},
            ]}
        ],
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

def main():
    # Paths
    problem= "disaster" #change this to "bummer", "glitch" or "disaster" as needed
    problem_c= problem.capitalize()
    image_dir = os.path.join(os.getcwd(),f"{problem_c}Folder")
 
    #input_file = os.path.join(image_dir, "Stats_summary_bummer_combined_1.csv")
    #output_file = os.path.join(image_dir, "Stats_summary_bummer_combined_cgpt_classify_image_1.csv")
    input_file=os.path.join(image_dir,f"Stats_summary_{problem}_combined.csv")
    output_file=f"Stats_summary_{problem}_combined_cgpt_classify_image.csv"
    df = pd.read_csv(input_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Initialize OpenAI client
    client = OpenAI()

    # Add columns if missing
    if "Image Path" not in df.columns:
        df["Image Path"] = ""
    if "Predicted Problem Size" not in df.columns:
        df.insert(8, "Predicted Problem Size", "")  # Insert column at index 8

    # Filter rows where "Image_Tool" is either "GPTimage" or "DallE3"
    df_filtered = df[df["Image_Tool"].isin(["GPTimage", "DallE3"])]

    # Loop through filtered rows
    for index, row in df_filtered.iterrows():
        tool = row["Image_Tool"]
        scenario = row["scenario"]  # Access the "scenario" column
        image_path = os.path.join(image_dir, f"scenario_{problem}_{scenario}_{tool}.png")
        
        if os.path.exists(image_path):
            try:
                predicted_size = predict_problem_size(client, image_path).lower()
                df.at[index, "Image Path"] = image_path
                df.at[index, "Predicted Problem Size"] = predicted_size
                print(f"[Scenario {scenario}] Tool: {tool}, Prediction: {predicted_size}")
            except Exception as e:
                print(f"[Scenario {scenario}] Tool: {tool}, Failed: {e}")
        else:
            print(f"[Scenario {scenario}] Tool: {tool}, Image not found: {image_path}")

    # Save result
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")

if __name__ == "__main__":
    main()