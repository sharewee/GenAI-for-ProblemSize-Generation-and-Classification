import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load all the keys from the .env file
load_dotenv()

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

def predict_problem_size(client: OpenAI, story: str) -> str:
    """
    Use ChatGPT to classify the size of the problem for a given story.
    """
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": story},
        ],
    )
    return response.output_text.strip().lower()

def main():
    problem= "glitch" #change this to "bummer", "glitch" or "disaster" as needed
    problem_c= problem.capitalize()
    image_dir = os.path.join(os.getcwd(),f"{problem_c}Folder")
    # Load the CSV file containing stories
    input_file=os.path.join(image_dir,f"Stats_summary_{problem}_combined.csv")
    output_file=f"Stats_summary_{problem}_combined_cgpt_classify_text.csv"
    df = pd.read_csv(input_file)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Verify column names
    print("Columns in the CSV file:", df.columns)

    # Initialize OpenAI client
    client = OpenAI()

    # Check if "Predicted Problem Size" column exists
    if "Predicted Problem Size" not in df.columns:
        # Insert the column next to "Problem Size"
        problem_size_index = df.columns.get_loc("Problem Size") + 1
        df.insert(problem_size_index, "Predicted Problem Size", "")  # Create the column if it doesn't exist
    
    # Process every other row
    processed_indices = []
    for index in range(0, len(df), 2):  # Skip every other row
        story = df.at[index, "Script"]  # Access the "Script" column
        scenario = df.at[index, "scenario"]  # Access the "scenario" column
        predicted_size = predict_problem_size(client, story)  # Predict problem size for each story
        print(f"[Scenario {scenario}] Prediction: {predicted_size}")  # Output the prediction with scenario
        df.at[index, "Predicted Problem Size"] = predicted_size  # Override or populate the column
        processed_indices.append(index)  # Track processed rows

    # Remove skipped rows
    df = df.loc[processed_indices].reset_index(drop=True)

    # Remove the "Image_Tool" column
    if "Image_Tool" in df.columns:
        df.drop(columns=["Image_Tool"], inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()