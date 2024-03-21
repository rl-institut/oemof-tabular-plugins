import os
import pytest
import src.oemof_tabular_plugins.general.pre_processing as gen_pre_p

test_inputs = os.path.join("tests", "test_inputs")

@pytest.mark.parametrize("capex, opex_fix, lifetime, wacc, expected_annuity", [
    (1000000, 50000, 20, 0.05, 130242.59),  # Adjust the expected_annuity based on your calculations
    # Add more test cases with different input values and their expected_annuity results
])
def test_calculate_annuity(capex, opex_fix, lifetime, wacc, expected_annuity):
    result = gen_pre_p.calculate_annuity(capex, opex_fix, lifetime, wacc)
    assert result == expected_annuity

def test_no_capacity_cost(test_inputs):


