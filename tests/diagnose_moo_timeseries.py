"""
Diagnostic script to understand why MOO with [1,0,0,0] produces different results than SOO
"""
import os
import pandas as pd
import numpy as np

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# After running the test, check the temporary scenario directories
# We'll manually inspect one scenario to see what was written

def check_preprocessed_files(scenario_dir, scenario_name="SOO"):
    """Check what values were written by preprocessing"""

    print(f"\n{'='*80}")
    print(f"ANALYZING {scenario_name} PREPROCESSED FILES")
    print(f"{'='*80}")

    # Check element files
    elements_dir = os.path.join(scenario_dir, "data", "elements")

    # Check a conversion component (e.g., inverter)
    conversion_path = os.path.join(elements_dir, "energy_conversion.csv")
    if os.path.exists(conversion_path):
        print(f"\n--- energy_conversion.csv ---")
        df = pd.read_csv(conversion_path, sep=";")
        cols = ['name', 'capacity_cost', 'marginal_cost', 'resource_cost', 'annuity']
        cols_present = [c for c in cols if c in df.columns]
        print(df[cols_present])

    # Check a dispatchable source (e.g., diesel)
    sources_path = os.path.join(elements_dir, "energy_sources.csv")
    if os.path.exists(sources_path):
        print(f"\n--- energy_sources.csv ---")
        df = pd.read_csv(sources_path, sep=";")
        cols = ['name', 'capacity_cost', 'marginal_cost', 'resource_cost']
        cols_present = [c for c in cols if c in df.columns]
        print(df[cols_present])

    # Check if time series profiles were created
    sequences_dir = os.path.join(scenario_dir, "data", "sequences")
    profiles_path = os.path.join(sequences_dir, "profiles.csv")

    if os.path.exists(profiles_path):
        print(f"\n--- Time Series Profiles ---")
        df_profiles = pd.read_csv(profiles_path, sep=";")

        # Look for MOO-generated profiles (they end with _mc_profile)
        moo_cols = [col for col in df_profiles.columns if col.endswith('_mc_profile')]

        if moo_cols:
            print(f"\nFound {len(moo_cols)} MOO-generated marginal cost profiles:")
            for col in moo_cols[:3]:  # Show first 3
                values = df_profiles[col].values
                print(f"\n{col}:")
                print(f"  First 5 values: {values[:5]}")
                print(f"  Min: {values.min():.6e}, Max: {values.max():.6e}, Mean: {values.mean():.6e}")
                print(f"  Std Dev: {values.std():.6e}")
                print(f"  All identical? {np.allclose(values, values[0], rtol=1e-10)}")
        else:
            print("No MOO-generated marginal cost profiles found (this is expected for SOO)")


def compare_marginal_costs(soo_dir, moo_dir):
    """Compare the marginal costs between SOO and MOO"""

    print(f"\n{'='*80}")
    print("COMPARING MARGINAL COSTS: SOO vs MOO")
    print(f"{'='*80}")

    # For energy_sources (dispatchable)
    soo_sources = pd.read_csv(os.path.join(soo_dir, "data/elements/energy_sources.csv"), sep=";")
    moo_sources = pd.read_csv(os.path.join(moo_dir, "data/elements/energy_sources.csv"), sep=";")

    print("\n--- Dispatchable Sources (energy_sources.csv) ---")
    for idx, row in soo_sources.iterrows():
        name = row['name']
        soo_mc = row['marginal_cost']
        moo_mc = moo_sources[moo_sources['name'] == name]['marginal_cost'].values[0]

        print(f"\n{name}:")
        print(f"  SOO marginal_cost: {soo_mc}")
        print(f"  MOO marginal_cost: {moo_mc}")

        if isinstance(moo_mc, str) and moo_mc.endswith('_mc_profile'):
            # MOO created a time series - check its values
            moo_profiles = pd.read_csv(os.path.join(moo_dir, "data/sequences/profiles.csv"), sep=";")
            if moo_mc in moo_profiles.columns:
                ts_values = moo_profiles[moo_mc].values
                print(f"  MOO time series stats:")
                print(f"    Length: {len(ts_values)}")
                print(f"    Min: {ts_values.min():.6e}, Max: {ts_values.max():.6e}")
                print(f"    Mean: {ts_values.mean():.6e}, Std: {ts_values.std():.6e}")
                print(f"    Constant? {np.allclose(ts_values, ts_values[0], rtol=1e-10)}")

                # If SOO marginal_cost is numeric, compare
                if isinstance(soo_mc, (int, float)):
                    print(f"  Expected value (from SOO): {soo_mc:.6e}")
                    print(f"  MOO time series mean: {ts_values.mean():.6e}")
                    print(f"  Difference: {abs(ts_values.mean() - soo_mc):.6e}")


if __name__ == "__main__":
    # You need to run this AFTER the test has run and BEFORE temp directories are deleted
    # Modify the test to not delete temp_dir, or manually specify directories here

    print("This script should be run after test_soo_moo_equivalence.py")
    print("Modify the test to keep the temp directories, then run this script.")
    print("\nExample usage after modifying test:")
    print("  soo_dir = '/path/to/temp/scenario_soo'")
    print("  moo_dir = '/path/to/temp/scenario_moo'")
    print("  check_preprocessed_files(soo_dir, 'SOO')")
    print("  check_preprocessed_files(moo_dir, 'MOO')")
    print("  compare_marginal_costs(soo_dir, moo_dir)")
