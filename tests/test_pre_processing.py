import os
import shutil

import pytest
from oemof_tabular_plugins.general import pre_processing, calculate_annuity

test_inputs = os.path.join("tests", "test_inputs_pre_processing")


@pytest.mark.parametrize("capex, opex_fix, lifetime, wacc, expected_annuity", [
     (1000000, 50000, 20, 0.05, 130242.59)
])
def test_calculate_annuity(capex, opex_fix, lifetime, wacc, expected_annuity):
    result = calculate_annuity(capex, opex_fix, lifetime, wacc)
    assert result == expected_annuity


class TestPreprocessing:
    test_inputs_pre_p = os.path.join("tests", "test_inputs_pre_processing")
    pre_p_dir = os.path.join("tests", "test_inputs_pre_processing", "pre_processing")
    _package_path = os.path.join("data", "elements")
    package_path = _package_path

    def setup_method(self):
        if os.path.exists(self.pre_p_dir):
            shutil.rmtree(self.pre_p_dir, ignore_errors=True)
        os.mkdir(self.pre_p_dir)
        os.mkdir(os.path.join(self.pre_p_dir, "data"))
        os.mkdir(os.path.join(self.pre_p_dir, self._package_path))
        self.package_path = os.path.join(self.pre_p_dir, self._package_path)

    def teardown_method(self):
        # if os.path.exists(self.test_inputs_pre_processing):
        #    shutil.rmtree(self.test_inputs_pre_processing, ignore_errors=True)
        self.package_path = self._package_path

    def test_scenario_a1_raises_error(self):
        f_name = "annuity_empty_no_cost_params.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)
        print('done')
