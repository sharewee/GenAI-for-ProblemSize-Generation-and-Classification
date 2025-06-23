import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Confusion matrix for problem size classification with text prompts

# List of CSV files to load
csv_files = [
    "Stats_summary_bummer_combined_cgpt_classify_text.csv",
    "Stats_summary_disaster_combined_cgpt_classify_text.csv",
    "Stats_summary_glitch_combined_cgpt_classify_text.csv"
]

# Load and combine data
dfs = [pd.read_csv(file)[["Problem Size", "Predicted Problem Size"]] for file in csv_files]
df_all = pd.concat(dfs, ignore_index=True)

# Define labels
labels = ["glitch", "bummer", "disaster"]

# Compute confusion matrix
cm = confusion_matrix(df_all["Problem Size"], df_all["Predicted Problem Size"], labels=labels)

# Normalize by row (true labels) to get percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

# Plot normalized confusion matrix (with numbers, no % sign)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=["Glitch", "Bummer", "Disaster"])
fig, ax = plt.subplots()
disp.plot(cmap='Blues', values_format=".2f", ax=ax)

# Set color scale limits manually
im = ax.images[0]  # Access the image object created by ConfusionMatrixDisplay
im.set_clim(0, 100)  # Set vmin=0 and vmax=100

plt.title("Confusion Matrix of Text Classified by ChatGPT (%)")
plt.show()

# Print matrix with percentage signs
cm_percentage_with_sign = np.array([[f"{value:.2f}%" for value in row] for row in cm_percentage])
print("Confusion Matrix with Percentage Signs:")
print(cm_percentage_with_sign)