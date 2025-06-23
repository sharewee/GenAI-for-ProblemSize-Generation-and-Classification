import os
import pandas as pd
import time
import google.generativeai as genai


genai.configure()

PROMPT = """
You will view a video telling a short story about a child experiencing a social problem. 
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

def classify_video(video_path):
    """Classify the problem size based on the video using Gemini API."""
    try:
        # Upload the video file
        myfile = genai.upload_file(
            path=video_path,
            display_name=os.path.basename(video_path)
        )
        print(f"Uploaded file '{os.path.basename(video_path)}' as: {myfile.uri}")

        # Add a fixed delay to allow the file to process
        print(f"Waiting for the file to process...")
        time.sleep(5)  # Wait for 5 seconds (adjust if necessary)

        # Use the file for classification
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content([
            myfile, PROMPT
        ])
        return response.text.strip().lower()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return "Error"

def main():
    # File paths

    problem= "glitch" #change this to "bummer", "glitch" or "disaster" as needed
    problem_c= problem.capitalize()
    video_dir = os.path.join(os.getcwd(),f"{problem_c}Folder")
    # Load the CSV file containing stories
    input_csv=os.path.join(video_dir,f"Stats_summary_{problem}_combined.csv")
    output_csv=f"Stats_summary_{problem}_combined_gemini_classify_video.csv"

  
    # Read CSV
    df = pd.read_csv(input_csv)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Add columns if missing
    if "Video Path" not in df.columns:
        df["Video Path"] = ""
    if "Predicted Problem Size" not in df.columns:
        df.insert(df.columns.get_loc("Problem Size") + 1, "Predicted Problem Size", "")

    # Loop through rows
    for index, row in df.iterrows():
        tool = row["Image_Tool"]
        scenario = row["scenario"]  # Access the "scenario" column
        video_path = os.path.join(video_dir, f"video_{problem}_{scenario}_{tool}.mp4")
        
        if os.path.exists(video_path):
            try:
                predicted_size = classify_video(video_path)
                df.at[index, "Video Path"] = video_path
                df.at[index, "Predicted Problem Size"] = predicted_size
                print(f"[Scenario {scenario}] Tool: {tool}, Prediction: {predicted_size}")
            except Exception as e:
                print(f"[Scenario {scenario}] Tool: {tool}, Failed: {e}")
        else:
            print(f"[Scenario {scenario}] Tool: {tool}, Video not found: {video_path}")

    # Save result
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")

if __name__ == "__main__":
    main()