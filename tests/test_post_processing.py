import os
import shutil
import pandas as pd
import pytest
from oemof_tabular_plugins.script import compute_scenario
from unittest.mock import patch
import json
from oemof.tabular.facades import TYPEMAP
from oemof_tabular_plugins.datapackage.building import infer_metadata_from_data
from oemof_tabular_plugins.wefe.facades import PVPanel, MIMO


TYPEMAP["pv-panel"] = PVPanel
TYPEMAP["mimo"] = MIMO


class TestPostProcessingBusCarrierMapping:
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
            skip_preprocessing=True,
            skip_infer_datapackage_metadata=True,
        )

    def teardown_method(self):
        """
        Cleans up the testing environment after each test method is executed.
        """
        # removes the 'pre_processing' directory if it exists, ignoring errors
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir, ignore_errors=True)

    def test_carrier_found_in_busses_logs_message(self, caplog):
        """
        The carriers are defined within a column 'carrier' of 'elements/bus.csv' file
        """
        f_name = "carriers_in_busses_only"
        infer_metadata_from_data(
            package_name=f_name,
            path=os.path.join(self.pre_p_dir, f_name),
            typemap=TYPEMAP,
        )
        # TODO add this assertion once oemof.tabular does not require carrier in its facade attributes (default to "")
        # calc = self.compute_scenario(f_name)
        # assert "carrier" in calc.df_results.index.names
        assert any(
            record.levelname == "INFO"
            and "Bus to carrier mapping found in 'elements/bus.csv' file of datapackage"
            in record.message
            for record in caplog.records
        )

    def test_carrier_found_in_busses_trumps_component_carriers(self, caplog):
        """
        If the carriers are defined within 'elements/bus.csv' file the carriers defined at the component level do not play a role
        """
        f_name = "carriers_over_defined"
        calc = self.compute_scenario(f_name)

        assert "carrier" in calc.df_results.index.names
        assert any(
            record.levelname == "INFO"
            and "Bus to carrier mapping found in 'elements/bus.csv' file of datapackage"
            in record.message
            for record in caplog.records
        )

    def test_carrier_inferred_from_components_carriers(self, caplog):
        """
        If the carriers are not defined within 'elements/bus.csv' file they will be inferred from the components
        """
        f_name = "carriers_in_components_only"
        calc = self.compute_scenario(f_name)

        assert "carrier" in calc.df_results.index.names
        assert any(
            record.levelname == "WARNING"
            and "the bus-carrier mapping will be inferred from the component's carrier"
            in record.message
            for record in caplog.records
        )

    def test_carrier_mapping_skipped(self, caplog):
        """
        If the carriers are not defined within 'elements/bus.csv' file and the user set infer_bus_carrier=False, then the results will simply not add a 'carrier' level in the MultiIndex of the results
        """
        f_name = "carriers_in_components_only"
        calc = self.compute_scenario(
            f_name,
            infer_bus_carrier=False,
        )
        assert "carrier" not in calc.df_results.index.names
        assert any(
            record.levelname == "INFO"
            and "Result dataframe will not contain 'carrier' in its MultiIndex levels."
            in record.message
            for record in caplog.records
        )

    def test_carrier_not_defined(self):
        """
        If the carriers are not defined within 'elements/bus.csv' file they will be inferred from the components, if the components themselves do not have carrier attributes an error is raised
        """
        f_name = "carriers_not_defined"
        with pytest.raises(ValueError) as e_info:
            infer_metadata_from_data(
                package_name=f_name,
                path=os.path.join(self.pre_p_dir, f_name),
                typemap=TYPEMAP,
            )

        assert "The bus-carrier mapping is empty" in str(e_info.value)

    def test_carrier_under_defined(self):
        """
        If the carriers are not defined within 'elements/bus.csv' file they will be inferred from the components, if the components do not have enough carrier attributes to map all busses an error is raised
        """
        f_name = "carriers_under_defined"
        with pytest.raises(ValueError) as e_info:
            infer_metadata_from_data(
                package_name=f_name,
                path=os.path.join(self.pre_p_dir, f_name),
                typemap=TYPEMAP,
            )

        assert (
            "Bus 'permeate-bus' is missing from the busses carrier dict inferred from the EnergySystem instance"
            in str(e_info.value)
        )

    def test_carrier_wrongly_defined(self):
        """
        If the carriers are not defined within 'elements/bus.csv' file they will be inferred from the components, if two components connected to the same bus have disctinct carriers an error is raised
        """
        f_name = "carriers_wrongly_defined"
        with pytest.raises(ValueError) as e_info:
            infer_metadata_from_data(
                package_name=f_name,
                path=os.path.join(self.pre_p_dir, f_name),
                typemap=TYPEMAP,
            )

        assert "are associated to the same bus" in str(e_info.value)
