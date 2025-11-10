"""
Test to verify that Single-Objective Optimization (SOO) and Multi-Objective Optimization (MOO)
with weight factors [1, 0, 0, 0] produce equivalent optimal solutions.

Key insight:
- During optimization:
  * SOO uses: capacity_cost = annuity, marginal_cost = resource_cost
  * MOO with [1,0,0,0] uses: capacity_cost = 10^15 * annuity / global_GDP
                             marginal_cost = 10^15 * resource_cost / global_GDP

- Since both are linear scalings of the same objective, they produce identical optimal capacities/flows

- During post-processing:
  * Costs are calculated from: capacity × original_annuity + flow × original_resource_cost
  * Both use the ORIGINAL (unscaled) cost parameters

Therefore, SOO and MOO with [1,0,0,0] should produce:
- Identical optimal capacities
- Identical flow patterns
- Identical absolute costs (post-processing uses original cost values, not scaled ones)

Implementation:
- Preprocessing is idempotent: always resets and recalculates costs from base parameters
- No git restore needed - preprocessing cleans up MOO artifacts from previous runs
"""
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest

from oemof_tabular_plugins.script.compute import compute_scenario
from oemof_tabular_plugins.wefe import WEFE_TYPEMAP as TYPEMAP


def test_soo_moo_cost_only_equivalence():
    """
    Test that SOO and MOO with weights [1,0,0,0] produce equivalent optimal solutions.
    """
    # Setup paths
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    scenario_name = "scenario_19"
    scenario_dir_original = os.path.join(project_dir, "examples", "scenarios", scenario_name)

    # Check if scenario exists
    if not os.path.exists(scenario_dir_original):
        pytest.skip(f"Scenario {scenario_name} not found at {scenario_dir_original}")

    # Create temporary directories for both scenario copies and results
    # Note: Preprocessing is now idempotent and will reset/recalculate costs from base parameters
    temp_dir = tempfile.mkdtemp()
    try:
        # Create separate scenario copies to ensure clean state for each run
        scenario_dir_soo = os.path.join(temp_dir, "scenario_soo")
        scenario_dir_moo = os.path.join(temp_dir, "scenario_moo")
        results_path_soo = os.path.join(temp_dir, "soo_results")
        results_path_moo = os.path.join(temp_dir, "moo_results")

        # Copy scenario directory twice for independent runs
        print(f"Copying scenario to temporary locations...")
        shutil.copytree(scenario_dir_original, scenario_dir_soo)
        shutil.copytree(scenario_dir_original, scenario_dir_moo)
        print(f"✓ Created independent scenario copies for SOO and MOO")

        # Common parameters
        wacc = 0.06
        custom_attributes = [
            "ghg_emission_factor",
            "renewable_factor",
            "land_requirement_factor",
            "water_consumption_factor",
            "indirect_water_consumption_factor",
            "land_requirement",
            "water_footprint",
            "ghg_emissions",
            "resource_cost",
            "annuity"
        ]

        # MOO weight factors - cost only
        moo_wf = {
            "wf_cost": 1,
            "wf_ghg": 0.,
            "wf_lr": 0.,
            "wf_wf": 0.,
        }

        print("\n" + "="*80)
        print("TEST: SOO vs MOO with cost-only weights [1,0,0,0]")
        print("="*80)

        print("\n" + "="*80)
        print("Step 1/3: Running SOO (moo=False)")
        print("="*80)
        calculator_soo = compute_scenario(
            scenario_dir_soo,
            results_path_soo,
            wacc,
            scenario_name=scenario_name,
            custom_attributes=custom_attributes,
            typemap=TYPEMAP,
            moo=False,
            moo_wf=None,
            dash_app=False,
            parameters_units=None,
            skip_infer_datapackage_metadata=True,
            save_raw_results=True,
        )
        print(f"✓ SOO completed. Results saved to {results_path_soo}")

        print("\n" + "="*80)
        print("Step 2/3: Running MOO with weights [1, 0, 0, 0] (cost only)")
        print("="*80)
        calculator_moo = compute_scenario(
            scenario_dir_moo,
            results_path_moo,
            wacc,
            scenario_name=scenario_name,
            custom_attributes=custom_attributes,
            typemap=TYPEMAP,
            moo=True,
            moo_wf=moo_wf,
            dash_app=False,
            parameters_units=None,
            skip_infer_datapackage_metadata=True,
            save_raw_results=True,
        )
        print(f"✓ MOO completed. Results saved to {results_path_moo}")

        print("\n" + "="*80)
        print("Step 3/3: COMPARING RESULTS")
        print("="*80)

        # ========================================
        # 1. Compare Capacities (should be identical)
        # ========================================
        print("\n" + "-"*80)
        print("1. COMPARING CAPACITIES")
        print("-"*80)

        df_soo = calculator_soo.df_results
        df_moo = calculator_moo.df_results

        # Extract capacity-related columns (only string columns)
        capacity_cols = [col for col in df_soo.columns
                        if isinstance(col, str) and ('capacity' in col.lower() or 'investment' in col.lower())]

        capacity_diffs = []
        for col in capacity_cols:
            if col in df_soo.columns and col in df_moo.columns:
                soo_vals = df_soo[col].fillna(0)
                moo_vals = df_moo[col].fillna(0)

                if not np.allclose(soo_vals, moo_vals, rtol=1e-4, atol=1e-6):
                    max_diff = np.abs(soo_vals - moo_vals).max()
                    capacity_diffs.append({
                        'column': col,
                        'max_difference': max_diff,
                        'soo_sum': soo_vals.sum(),
                        'moo_sum': moo_vals.sum()
                    })
                    print(f"  ⚠ {col}: max difference = {max_diff:.6e}")
                else:
                    print(f"  ✓ {col}: identical")

        if capacity_diffs:
            print(f"\n⚠ WARNING: {len(capacity_diffs)} capacity columns differ!")
            for diff in capacity_diffs:
                print(f"  - {diff['column']}: max_diff = {diff['max_difference']:.6e}")
            assert False, "Capacities should be identical for SOO and MOO with cost-only weights!"
        else:
            print("\n✓ All capacity values are identical!")

        # ========================================
        # 2. Compare Flow Patterns
        # ========================================
        print("\n" + "-"*80)
        print("2. COMPARING FLOW PATTERNS")
        print("-"*80)

        # Extract flow columns (sequences, profiles, etc.) - only string columns
        flow_cols = [col for col in df_soo.columns
                    if isinstance(col, str) and ('flow' in col.lower() or 'sequence' in col.lower())]

        flow_diffs = []
        for col in flow_cols[:10]:  # Limit to first 10 for brevity
            if col in df_soo.columns and col in df_moo.columns:
                soo_vals = df_soo[col].fillna(0)
                moo_vals = df_moo[col].fillna(0)

                if not np.allclose(soo_vals, moo_vals, rtol=1e-4, atol=1e-6):
                    max_diff = np.abs(soo_vals - moo_vals).max()
                    flow_diffs.append(col)
                    print(f"  ⚠ {col}: max difference = {max_diff:.6e}")
                else:
                    print(f"  ✓ {col}: identical")

        if flow_diffs:
            print(f"\n⚠ {len(flow_diffs)} flow patterns differ (this may indicate an issue)")
        else:
            print("\n✓ Flow patterns are identical!")

        # ========================================
        # 3. Compare Absolute Costs (should be identical!)
        # ========================================
        print("\n" + "-"*80)
        print("3. COMPARING ABSOLUTE COSTS")
        print("-"*80)
        print("Since post-processing uses original annuity/resource_cost values")
        print("(not the scaled MOO optimization values), absolute costs should be identical!")

        # Get cost-related outputs
        soo_costs = {k: v for k, v in calculator_soo.calculated_outputs.items() if 'cost' in k.lower() and isinstance(v, (int, float))}
        moo_costs = {k: v for k, v in calculator_moo.calculated_outputs.items() if 'cost' in k.lower() and isinstance(v, (int, float))}

        print("\nCost comparison:")
        cost_diffs = []
        for key in soo_costs.keys():
            if key in moo_costs:
                soo_val = soo_costs[key]
                moo_val = moo_costs[key]
                diff = abs(soo_val - moo_val)
                rel_diff = diff / max(abs(soo_val), 1e-10) * 100  # Relative difference in %

                match = "✓" if diff < 1e-6 else "✗"
                print(f"  {match} {key:40s}: SOO={soo_val:15.2f}, MOO={moo_val:15.2f}, diff={diff:.2e} ({rel_diff:.2f}%)")

                if diff > 1e-4:  # Significant difference threshold
                    cost_diffs.append({
                        'key': key,
                        'soo': soo_val,
                        'moo': moo_val,
                        'diff': diff,
                        'rel_diff': rel_diff
                    })

        if cost_diffs:
            print(f"\n⚠ WARNING: {len(cost_diffs)} cost values differ significantly!")
            for diff in cost_diffs:
                print(f"  - {diff['key']}: diff = {diff['diff']:.2e} ({diff['rel_diff']:.2f}%)")
        else:
            print("\n✓ All absolute costs are identical!")

        # ========================================
        # 4. Summary
        # ========================================
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        all_passed = len(capacity_diffs) == 0 and len(flow_diffs) == 0 and len(cost_diffs) == 0

        if all_passed:
            print("✓ TEST PASSED: SOO and MOO with [1,0,0,0] weights produce equivalent solutions")
            print("  - Optimal capacities are identical")
            print("  - Flow patterns are identical")
            print("  - Absolute costs are identical (post-processing uses original cost values)")
        else:
            print("✗ TEST FAILED: Differences detected")
            if capacity_diffs:
                print(f"  - {len(capacity_diffs)} capacity columns differ")
            if flow_diffs:
                print(f"  - {len(flow_diffs)} flow patterns differ")
            if cost_diffs:
                print(f"  - {len(cost_diffs)} cost values differ")

        assert all_passed, "SOO and MOO should produce identical results when weights are [1,0,0,0]"

    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\n✓ Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    test_soo_moo_cost_only_equivalence()
