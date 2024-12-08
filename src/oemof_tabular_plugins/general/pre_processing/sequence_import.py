import pandas as pd
import os
# Print the current working directory
print("Current working directory:", os.getcwd())
# Current working directory: C:\Users\jufle\dev\oemof-tabular-plugins\src\oemof_tabular_plugins\general\pre_processing
# Path to the CSV file (adjusted relative path)
#TODO: make path inspecific to scenario/avoid hard-coding
csv_file_path = "../../examples/scenarios/aiwa_24/data/sequences/volatile_profile.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Display the first few rows of the DataFrame
print(df.head())
# Check if the 'cf_aware' column exists and print it
if 'cf_aware' in df.columns:
    print(df['cf_aware'])
else:
    print("Column 'cf_aware' does not exist in the DataFrame.")
    