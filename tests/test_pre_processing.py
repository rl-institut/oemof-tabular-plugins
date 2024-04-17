import os
import shutil

import pandas as pd
import pytest
from oemof_tabular_plugins.general import pre_processing, calculate_annuity
from unittest.mock import patch


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

    def test_annuity_empty_no_cost_params_raises_error(self):
        f_name = "annuity_empty_no_cost_params.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)

    def test_annuity_defined_no_cost_params_logs_message(self, caplog):
        f_name = "annuity_defined_no_cost_params.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        pre_processing(self.pre_p_dir, wacc=1)
        # check if the info message is logged
        assert any(record.levelname == "INFO" and "The annuity cost is directly used for" in record.message
                   for record in caplog.records)

    def test_annuity_defined_no_cost_params_uses_annuity(self):
        f_name = "annuity_defined_no_cost_params.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        pre_processing(self.pre_p_dir, wacc=1)
        expected_value = 107265
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=';')
        actual_value = df['capacity_cost'].iloc[0]
        assert actual_value == expected_value

    def test_annuity_empty_partial_cost_params_raises_error(self):
        f_name = "annuity_empty_partial_cost_params.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)

    def test_annuity_defined_partial_cost_params_raises_error(self):
        f_name = "annuity_defined_partial_cost_params.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)

    def test_annuity_empty_all_cost_params_defined_logs_message(self, caplog):
        f_name = "annuity_empty_all_cost_params_defined.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        pre_processing(self.pre_p_dir, wacc=1)
        # check if the info message is logged
        assert any(record.levelname == "INFO" and "has been calculated and updated" in record.message
                   for record in caplog.records)

    def test_annuity_empty_all_cost_params_defined_calculates_annuity(self):
        f_name = "annuity_empty_all_cost_params_defined.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        wacc = 1
        pre_processing(self.pre_p_dir, wacc=wacc)
        expected_value = calculate_annuity(capex=975000, opex_fix=11625, lifetime=20, wacc=wacc)
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=';')
        actual_value = df['capacity_cost'].iloc[0]
        assert actual_value == expected_value

    def test_annuity_all_cost_params_some_empty_raises_error(self):
        f_name = "annuity_all_cost_params_some_empty.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)

    def test_annuity_defined_all_cost_params_defined_yes_calculates_new_annuity(self):
        f_name = "annuity_defined_all_cost_params_defined.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        wacc = 1
        # patch the input() function to return 'yes' during the test
        with patch('builtins.input', return_value='yes'):
            pre_processing(self.pre_p_dir, wacc=wacc)
            expected_value = calculate_annuity(capex=975000, opex_fix=11625, lifetime=20, wacc=1)
            df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=';')
            actual_value = df['capacity_cost'].iloc[0]
        assert actual_value == expected_value

    def test_annuity_defined_all_cost_params_defined_yes_uses_old_annuity(self):
        f_name = "annuity_defined_all_cost_params_defined.csv"
        shutil.copy(os.path.join(self.test_inputs_pre_p, f_name), os.path.join(self.package_path, f_name))
        wacc = 1
        # patch the input() function to return 'no' during the test
        with patch('builtins.input', return_value='no'):
            pre_processing(self.pre_p_dir, wacc=wacc)
            expected_value = 107265
            df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=';')
            actual_value = df['capacity_cost'].iloc[0]
        assert actual_value == expected_value
