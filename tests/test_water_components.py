import os
import shutil

from oemof_tabular_plugins.script import compute_scenario
from oemof_tabular_plugins.datapackage.building import infer_metadata_from_data
from oemof_tabular_plugins.wefe import WEFE_TYPEMAP as TYPEMAP


class TestWaterComponents:
    """
    This class contains tests for the possible ways to use (or not) the mapping of bus to carrier in the results dataframe
    """

    # store the relevant paths
    test_inputs_path = os.path.join("tests", "test_inputs")
    pre_p_dir = os.path.join(
        "tests",
        "test_inputs",
    )
    results_dir = os.path.join("tests", "tests_outputs")
    _package_path = os.path.join("data", "elements")
    package_path = _package_path

    def setup_method(self):
        """
        Sets up the testing environment before each test method is executed.
        """
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir, ignore_errors=True)
        os.mkdir(self.results_dir)

    def compute_scenario(self, fname, infer_bus_carrier=True):
        """
        Run a single scenario and return a calculator instance
        """
        return compute_scenario(
            os.path.join(self.pre_p_dir, fname),
            self.results_dir,
            wacc=1,
            typemap=TYPEMAP,
            infer_bus_carrier=infer_bus_carrier,
            skip_preprocessing=False,
            skip_infer_datapackage_metadata=True,
        )

    def teardown_method(self):
        """
        Cleans up the testing environment after each test method is executed.
        """
        # removes the 'pre_processing' directory if it exists, ignoring errors
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir, ignore_errors=True)

    def test_run_water_component_test_through(self):
        """
        The carriers are defined within a column 'carrier' of 'elements/bus.csv' file
        """
        f_name = "water_components"

        calc = self.compute_scenario(f_name)
        assert "carrier" in calc.df_results.index.names
