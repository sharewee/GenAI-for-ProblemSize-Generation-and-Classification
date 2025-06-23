import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of CSV files for ChatGPT
cgpt_csv_files = [
    "Stats_summary_bummer_combined_cgpt_classify_image.csv",
    "Stats_summary_disaster_combined_cgpt_classify_image.csv",
    "Stats_summary_glitch_combined_cgpt_classify_image.csv"
]

# List of CSV files for Gemini
gemini_csv_files = [
    "Stats_summary_bummer_combined_gemini_classify_image.csv",
    "Stats_summary_disaster_combined_gemini_classify_image.csv",
    "Stats_summary_glitch_combined_gemini_classify_image.csv"
]

# Load and combine ChatGPT data
cgpt_dfs = [pd.read_csv(file) for file in cgpt_csv_files]
cgpt_df = pd.concat(cgpt_dfs, ignore_index=True)

# Filter rows where "Image_Tool" is "GPTimage"
cgpt_df = cgpt_df.loc[cgpt_df["Image_Tool"] == "GPTimage"]

cgpt_df["Correctness"] = cgpt_df["Problem Size"] == cgpt_df["Predicted Problem Size"]
cgpt_df["Correctness"] = cgpt_df["Correctness"].map({True: "Correct", False: "Incorrect"})
cgpt_df["Modality"] = cgpt_df["Correctness"].map(
    {"Correct": "Correct (by GPT-4o)", "Incorrect": "Incorrect (by GPT-4o)"}
)

# Load and combine Gemini data
gemini_dfs = [pd.read_csv(file) for file in gemini_csv_files]
gemini_df = pd.concat(gemini_dfs, ignore_index=True)

# Filter rows where "Image_Tool" is "GPTimage"
gemini_df = gemini_df.loc[gemini_df["Image_Tool"] == "GPTimage"]

gemini_df["Correctness"] = gemini_df["Problem Size"] == gemini_df["Predicted Problem Size"]
gemini_df["Correctness"] = gemini_df["Correctness"].map({True: "Correct", False: "Incorrect"})
gemini_df["Modality"] = gemini_df["Correctness"].map(
    {"Correct": "Correct (by Gemini)", "Incorrect": "Incorrect (by Gemini)"}
)

# Combine both datasets
cgpt_df["Time"] = pd.to_numeric(cgpt_df["Time_Image"], errors="coerce")  # Ensure numeric values
gemini_df["Time"] = pd.to_numeric(gemini_df["Time_Image"], errors="coerce")  # Ensure numeric values
combined_df = pd.concat([cgpt_df[["Modality", "Time"]], gemini_df[["Modality", "Time"]]], ignore_index=True)

# Calculate statistics for each modality
modalities = combined_df["Modality"].unique()
for modality in modalities:
    modality_data = combined_df[combined_df["Modality"] == modality]["Time"]
    median = modality_data.median()
    q1 = modality_data.quantile(0.25)  # 25th percentile
    q3 = modality_data.quantile(0.75)  # 75th percentile
    iqr = q3 - q1  # Interquartile range
    outliers = modality_data[(modality_data < q1 - 1.5 * iqr) | (modality_data > q3 + 1.5 * iqr)]
    
    # Print statistics to console
    print(f"Statistics for {modality}:")
    print(f"  Median: {median}")
    print(f"  25th Percentile (Q1): {q1}")
    print(f"  75th Percentile (Q3): {q3}")
    print(f"  Interquartile Range (IQR): {iqr}")
    print(f"  Outliers: {list(outliers)}")
    print()

# Create the box plot
plt.figure(figsize=(10, 7))
sns.boxplot(
    x="Modality",
    y="Time",
    data=combined_df,
    palette={
        "Correct (by GPT-4o)": "#5286ff",
        "Incorrect (by GPT-4o)": "#c7d7fa",
        "Correct (by Gemini)": "#5286ff",
        "Incorrect (by Gemini)": "#c7d7fa"
    },
    order=["Correct (by GPT-4o)", "Incorrect (by GPT-4o)", "Correct (by Gemini)", "Incorrect (by Gemini)"]
)

# Debugging: Print sample sizes
print("Sample sizes for each modality:")
print(combined_df["Modality"].value_counts())

# Calculate sample sizes explicitly
sample_sizes = {
    modality: len(combined_df[combined_df["Modality"] == modality])
    for modality in ["Correct (by GPT-4o)", "Incorrect (by GPT-4o)", "Correct (by Gemini)", "Incorrect (by Gemini)"]
}

# Set the y-axis limit
plt.ylim(10, 70)  # Set the y-axis maximum to 70

# Annotate sample sizes on the plot
order = ["Correct (by GPT-4o)", "Incorrect (by GPT-4o)", "Correct (by Gemini)", "Incorrect (by Gemini)"]
for modality, count in sample_sizes.items():
    plt.text(
        x=order.index(modality),  # Ensure alignment with the box plot order
        y=67,  # Place the text slightly above the y-axis maximum
        s=f"n = {count}",
        ha="center",
        fontsize=10,
        color="black"
    )

# Add labels and title
plt.title("GPT-4o Image Generation Time by Classification Result")
plt.ylabel("Time (seconds)")
plt.xlabel("Classification Result")

# Show the plot
plt.show()