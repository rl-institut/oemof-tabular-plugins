"""
Quick script to check what values are written by pre-processing for SOO vs MOO
"""
import pandas as pd
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
scenario_name = "scenario_19"
scenario_dir = os.path.join(project_dir, "examples", "scenarios", scenario_name)

# Check a specific element file
element_file = "pv_panel.csv"
element_path = os.path.join(scenario_dir, "data", "elements", element_file)

print(f"Reading: {element_path}\n")
df = pd.read_csv(element_path, sep=";")

# Show key cost columns
cols_to_show = ['name', 'capacity_cost', 'marginal_cost', 'resource_cost', 'annuity', 'capex', 'opex_fix', 'lifetime']
cols_present = [c for c in cols_to_show if c in df.columns]

print(df[cols_present])
