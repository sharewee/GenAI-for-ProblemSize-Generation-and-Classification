import os
import pandas as pd
import google.generativeai as genai

# Load API key
genai.configure()

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

def classify_image(image_path):
    try:
        sample_file = genai.upload_file(
            path=image_path,
            display_name=os.path.basename(image_path)
        )
        print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        response = model.generate_content([
            sample_file, PROMPT
        ])
        return response.text.strip().lower()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Error"

def main():
    # File paths
    problem= "disaster" #change this to "bummer", "glitch" or "disaster" as needed
    problem_c= problem.capitalize()
    image_dir = os.path.join(os.getcwd(),f"{problem_c}Folder")
    # Load the CSV file containing stories
    input_csv=os.path.join(image_dir,f"Stats_summary_{problem}_combined.csv")
    output_csv=f"Stats_summary_{problem}_combined_gemini_classify_image.csv"
    # Read CSV
    df = pd.read_csv(input_csv)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Add columns if missing
    if "Image Path" not in df.columns:
        df["Image Path"] = ""
    if "Predicted Problem Size" not in df.columns:
        df.insert(df.columns.get_loc("Problem Size") + 1, "Predicted Problem Size", "")

    # Flter rows where "Image_Tool" is either "GPTimage" or "DallE3"
    df_filtered = df[df["Image_Tool"].isin(["GPTimage", "DallE3"])]

    # Loop through filtered rows
    for index, row in df_filtered.iterrows():
        tool = row["Image_Tool"]
        scenario = row["scenario"]  # Access the "scenario" column
        image_path = os.path.join(image_dir, f"scenario_{problem}_{scenario}_{tool}.png")
        
        if os.path.exists(image_path):
            try:
                predicted_size = classify_image(image_path)
                df.at[index, "Image Path"] = image_path
                df.at[index, "Predicted Problem Size"] = predicted_size
                print(f"[Scenario {scenario}] Tool: {tool}, Prediction: {predicted_size}")
            except Exception as e:
                print(f"[Scenario {scenario}] Tool: {tool}, Failed: {e}")
        else:
            print(f"[Scenario {scenario}] Tool: {tool}, Image not found: {image_path}")

    # Save result
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")

if __name__ == "__main__":
    main()