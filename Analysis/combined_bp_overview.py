import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of CSV files to load
csv_files = [
    "Stats_summary_bummer_combined.csv",
    "Stats_summary_disaster_combined.csv",
    "Stats_summary_glitch_combined.csv"
]

# Load and combine data
dfs = [pd.read_csv(file) for file in csv_files]
df_all = pd.concat(dfs, ignore_index=True)

# Box #1: Every other line in the "Script" column time
script_times = df_all.loc[::2, "Time_Script"]  # Every other row
script_times = pd.to_numeric(script_times, errors="coerce").dropna()  # Convert to numeric and drop NaN
script_data = pd.DataFrame({"Modality": "Script", "Time": script_times})

# Box #2: "Time_Image" where "Image_Tool" is "DallE3"
dalle3_times = df_all.loc[df_all["Image_Tool"] == "DallE3", "Time_Image"]
dalle3_times = pd.to_numeric(dalle3_times, errors="coerce").dropna()  # Convert to numeric and drop NaN
dalle3_data = pd.DataFrame({"Modality": "DALL-E 3 Image", "Time": dalle3_times})

# Box #3: "Time_Image" where "Image_Tool" is "GPTimage"
gptimage_times = df_all.loc[df_all["Image_Tool"] == "GPTimage", "Time_Image"]
gptimage_times = pd.to_numeric(gptimage_times, errors="coerce").dropna()  # Convert to numeric and drop NaN
gptimage_data = pd.DataFrame({"Modality": "GPT-4o Image", "Time": gptimage_times})

# Combine all data
combined_data = pd.concat([script_data, dalle3_data, gptimage_data], ignore_index=True)

# Add sample size to modality labels with a newline
sample_sizes = combined_data["Modality"].value_counts()
combined_data["Modality"] = combined_data["Modality"].map(
    lambda x: f"{x}\n(n={sample_sizes[x]})"  # Add newline before sample size
)

# Dynamically generate the palette
palette = {
    f"Script\n(n={sample_sizes['Script']})": "#dc92e4",
    f"DALL-E 3 Image\n(n={sample_sizes['DALL-E 3 Image']})": "#ff723a",
    f"GPT-4o Image\n(n={sample_sizes['GPT-4o Image']})": "#7aa2ff"
}

# Calculate statistics for each modality
modalities = combined_data["Modality"].unique()
for modality in modalities:
    modality_data = combined_data[combined_data["Modality"] == modality]["Time"]
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
plt.figure(figsize=(8, 7))
sns.boxplot(
    x="Modality",
    y="Time",
    data=combined_data,
    palette=palette  # Use dynamically generated palette
)

# Add labels and title
plt.title("Generation Time Comparison across Modalities")
plt.ylabel("Time (seconds)")
plt.xlabel("Modality")

# Show the plot
plt.show()