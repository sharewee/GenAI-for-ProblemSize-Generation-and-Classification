import os
import pandas as pd
import google.generativeai as genai

# Load API key

genai.configure()

PROMPT = """
You will read a short story about a child experiencing a social problem. 
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

def classify_text(script_text):
    """Classify the problem size based on the text using Gemini API."""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        response = model.generate_content([
            {"text": PROMPT},  # System prompt
            {"text": script_text}  # User input
        ])
        return response.text.strip().lower()
    except Exception as e:
        print(f"Error processing script: {e}")
        return "Error"

def main():
    # File paths
    problem= "disaster" #change this to "bummer", "glitch" or "disaster" as needed
    problem_c= problem.capitalize()
    image_dir = os.path.join(os.getcwd(),f"{problem_c}Folder")
    # Load the CSV file containing stories
    input_csv=os.path.join(image_dir,f"Stats_summary_{problem}_combined.csv")
    output_csv=f"Stats_summary_{problem}_combined_gemini_classify_text.csv"
  

    # Read CSV
    df = pd.read_csv(input_csv)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Verify column names
    print("Columns in the CSV file:", df.columns)

    # Add columns if missing
    if "Predicted Problem Size" not in df.columns:
        df.insert(df.columns.get_loc("Problem Size") + 1, "Predicted Problem Size", "")

    # Keep track of processed rows
    processed_indices = []

    # Loop through every other row in the original DataFrame
    for index in range(0, len(df), 2):  # Skip every other row
        row = df.iloc[index]  # Access the row using iloc
        script_text = row["Script"]  # Access the "Script" column
        scenario = row["scenario"]  # Access the "scenario" column
        
        if pd.notna(script_text):  # Ensure the script text is valid
            try:
                predicted_size = classify_text(script_text)  # Classify the text
                df.at[index, "Predicted Problem Size"] = predicted_size
                print(f"[Scenario {scenario}] Prediction: {predicted_size}")
                processed_indices.append(index)  # Track the processed row
            except Exception as e:
                print(f"[Scenario {scenario}] Failed: {e}")
        else:
            print(f"[Scenario {scenario}] Script is empty or invalid.")

    # Keep only the processed rows
    df = df.loc[processed_indices].reset_index(drop=True)

    # Remove the "Image_Tool" column if it exists
    if "Image_Tool" in df.columns:
        df.drop(columns=["Image_Tool"], inplace=True)

    # Save result
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")

if __name__ == "__main__":
    main()