import os
import shutil
import pandas as pd
import pytest
from oemof_tabular_plugins.general import pre_processing, calculate_annuity
from unittest.mock import patch
import json


@pytest.mark.parametrize(
    "capex, opex_fix, lifetime, wacc, expected_annuity",
    [(1000000, 50000, 20, 0.05, 130242.59)],
)
def test_calculate_annuity(capex, opex_fix, lifetime, wacc, expected_annuity):
    """
    Tests that the calculate_annuity function works as expected.
    :param capex: CAPEX (currency/MW*) *or the unit you choose to use throughout the model e.g. kW/GW
    :param opex_fix: the fixed OPEX (currency/MW*/year)
    :param lifetime: the lifetime of the component (years)
    :param wacc: the weighted average cost of capital (WACC) applied throughout the model (%)
    :param expected_annuity: the expected annuity value (currency)
    """
    # stores actual result from the calculate_annuity function
    result = calculate_annuity(capex, opex_fix, lifetime, wacc)
    assert result == expected_annuity


class TestPreprocessingCosts:
    """
    This class contains tests for the pre-processing costs function.
    """

    # store the relevant paths
    test_inputs_pre_p = os.path.join("tests", "test_inputs_pre_processing_costs")
    pre_p_dir = os.path.join(
        "tests", "test_inputs_pre_processing_costs", "pre_processing"
    )
    _package_path = os.path.join("data", "elements")
    package_path = _package_path

    def setup_method(self):
        """
        Sets up the testing environment before each test method is executed.
        """
        # removes the 'pre_processing' directory if it exists, ignoring errors
        if os.path.exists(self.pre_p_dir):
            shutil.rmtree(self.pre_p_dir, ignore_errors=True)
        # creates the 'pre_processing' directory
        os.mkdir(self.pre_p_dir)
        # creates the 'data' directory within the 'pre_processing' directory
        os.mkdir(os.path.join(self.pre_p_dir, "data"))
        # creates the 'elements' directory within the 'data' directory
        os.mkdir(os.path.join(self.pre_p_dir, self._package_path))
        # sets the package path for the test environment
        self.package_path = os.path.join(self.pre_p_dir, self._package_path)

    def copy_file_to_package_path(self, f_name):
        """
        Copies the scenario CSV file to the package path
        :param f_name: scenario CSV filename
        """
        # copy scenario csv file and datapackage json file to the package path
        shutil.copy(
            os.path.join(self.test_inputs_pre_p, f_name),
            os.path.join(self.package_path, f_name),
        )

    def teardown_method(self):
        """
        Cleans up the testing environment after each test method is executed.
        """
        # removes the 'pre_processing' directory if it exists, ignoring errors
        if os.path.exists(self.pre_p_dir):
            shutil.rmtree(self.pre_p_dir, ignore_errors=True)
        # resets the package path to its original value
        self.package_path = self._package_path

    def test_annuity_empty_no_cost_params_raises_error(self):
        """
        Tests that a value error is raised when the annuity is left empty and the
        other cost parameters are not included.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_empty_no_cost_params.csv")
        # check if calling the pre_processing function with the given scenario raises a value error
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)

    def test_annuity_defined_no_cost_params_uses_annuity(self):
        """
        Tests that the annuity is directly used when it is defined and the other cost
        parameters are not included.
        """
        # copy scenario csv file to the package path
        f_name = "annuity_defined_no_cost_params.csv"
        self.copy_file_to_package_path(f_name)
        # call pre_processing function
        pre_processing(self.pre_p_dir, wacc=1)
        # check if the actual value matches the expected value
        expected_value = 107265
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
        actual_value = df["capacity_cost"].iloc[0]
        assert actual_value == expected_value

    def test_annuity_defined_no_cost_params_logs_message(self, caplog):
        """
        Tests that an info message is logged when the annuity is directly used and the other cost
        parameters are not included.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_defined_no_cost_params.csv")
        # check if the info message is logged when the pre_processing function is called
        pre_processing(self.pre_p_dir, wacc=1)
        assert any(
            record.levelname == "INFO"
            and "The annuity cost is directly used for" in record.message
            for record in caplog.records
        )

    def test_annuity_empty_partial_cost_params_raises_error(self):
        """
        Tests that a value error is raised when the annuity is left empty and not all other cost
        parameters are included.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_empty_partial_cost_params.csv")
        # check if calling the pre_processing function with the given scenario raises a value error
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)

    def test_annuity_defined_partial_cost_params_logs_message(self, caplog):
        """
        Tests that a warning message is logged when the annuity is defined and some but not all other cost
        parameters are included.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_defined_partial_cost_params.csv")
        pre_processing(self.pre_p_dir, wacc=1)
        assert any(
            record.levelname == "WARNING"
            and "directly used but be aware that some cost" in record.message
            for record in caplog.records
        )

    def test_annuity_defined_partial_cost_params_uses_annuity(self):
        """
        Tests that the annuity is directly used when it is defined and some but not all of the other cost
        parameters are included.
        """
        # copy scenario csv file to the package path
        f_name = "annuity_defined_partial_cost_params.csv"
        self.copy_file_to_package_path(f_name)
        # call pre_processing function
        pre_processing(self.pre_p_dir, wacc=1)
        # check if the actual value matches the expected value
        expected_value = 107265
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
        actual_value = df["capacity_cost"].iloc[0]
        assert actual_value == expected_value

    def test_annuity_empty_all_cost_params_defined_calculates_annuity(self):
        """
        Tests that the annuity is calculated and added to 'capacity_cost' when the annuity is left
        empty and all other cost parameters are defined.
        """
        # copy scenario csv file to the package path
        f_name = "annuity_empty_all_cost_params_defined.csv"
        self.copy_file_to_package_path(f_name)
        # call the pre_processing function with wacc = 1
        wacc = 1
        pre_processing(self.pre_p_dir, wacc=wacc)
        # check if the actual value matches the expected value
        expected_value = calculate_annuity(
            capex=975000, opex_fix=11625, lifetime=20, wacc=wacc
        )
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
        actual_value = df["capacity_cost"].iloc[0]
        assert actual_value == expected_value

    def test_annuity_empty_all_cost_params_defined_logs_message(self, caplog):
        """
        Tests that an info message is logged when a new annuity value is calculated and updated
        from the other cost parameters.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_empty_all_cost_params_defined.csv")
        # check if the info message is logged when the pre_processing function is called
        pre_processing(self.pre_p_dir, wacc=1)
        assert any(
            record.levelname == "INFO"
            and "has been calculated and updated" in record.message
            for record in caplog.records
        )

    def test_annuity_all_cost_params_some_empty_logs_message(self, caplog):
        """
        Tests that warning message is logged when some but not all of the other cost parameters are defined.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_all_cost_params_some_empty.csv")
        # check if the info message is logged when the pre_processing function is called
        pre_processing(self.pre_p_dir, wacc=1)
        assert any(
            record.levelname == "WARNING"
            and "annuity will be directly used but be aware" in record.message
            for record in caplog.records
        )

    def test_annuity_all_cost_params_some_empty_uses_annuity(self):
        """
        Tests that the annuity is directly used when it is defined and some but not all of the other cost
        parameters are defined.
        """
        # copy scenario csv file to the package path
        f_name = "annuity_all_cost_params_some_empty.csv"
        self.copy_file_to_package_path(f_name)
        # call pre_processing function
        pre_processing(self.pre_p_dir, wacc=1)
        # check if the actual value matches the expected value
        expected_value = 107265
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
        actual_value = df["capacity_cost"].iloc[0]
        assert actual_value == expected_value

    def test_annuity_defined_all_cost_params_defined_yes_calculates_new_annuity(self):
        """
        Tests that when the annuity is defined and all other cost parameters, if the user enters 'yes', the
        new annuity is calculated and replaces the old annuity.
        """
        # copy scenario csv file to the package path
        f_name = "annuity_defined_all_cost_params_defined.csv"
        self.copy_file_to_package_path(f_name)
        wacc = 1
        # patch the input() function to return 'yes' during the test
        with patch("builtins.input", return_value="yes"):
            # call the pre_processing function with wacc = 1
            pre_processing(self.pre_p_dir, wacc=wacc)
            # check if the actual value matches the expected value
            expected_value = calculate_annuity(
                capex=975000, opex_fix=11625, lifetime=20, wacc=1
            )
            df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
            actual_value = df["capacity_cost"].iloc[0]
        assert actual_value == expected_value

    def test_annuity_defined_all_cost_params_defined_no_uses_old_annuity(self):
        """
        Tests that when the annuity is defined and all other cost parameters, if the user enters 'no', the
        old annuity is used.
        """
        # copy scenario csv file to the package path
        f_name = "annuity_defined_all_cost_params_defined.csv"
        self.copy_file_to_package_path(f_name)
        wacc = 1
        # patch the input() function to return 'no' during the test
        with patch("builtins.input", return_value="no"):
            # call the pre_processing function with wacc = 1
            pre_processing(self.pre_p_dir, wacc=wacc)
            # check if the actual value matches the expected value
            expected_value = 107265
            df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
            actual_value = df["capacity_cost"].iloc[0]
        assert actual_value == expected_value

    def test_annuity_defined_all_cost_params_defined_no_logs_message(self, caplog):
        """
        Tests that a warning message is logged when the annuity is used and the other cost
        parameters are defined but not used after the user enters 'no'.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_defined_all_cost_params_defined.csv")
        wacc = 1
        # patch the input() function to return 'no' during the test
        with patch("builtins.input", return_value="no"):
            # check if the warning message is logged when the pre_processing function is called and the user
            # enters 'no'
            pre_processing(self.pre_p_dir, wacc=wacc)
        assert any(
            record.levelname == "WARNING"
            and "could lead to discrepancies in the results" in record.message
            for record in caplog.records
        )

    def test_annuity_defined_all_cost_params_defined_invalid_input_retries(
        self, caplog
    ):
        """
        Tests that the user is asked to re-enter a valid input when the user enters an invalid input.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("annuity_defined_all_cost_params_defined.csv")
        wacc = 1
        # patch the input() function to return an invalid input ('invalid', in this case) during the first call,
        # and 'no' during the second call
        with patch("builtins.input", side_effect=["invalid", "no"]):
            # call the pre_processing function with wacc = 1
            pre_processing(self.pre_p_dir, wacc=wacc)
        # check if the invalid log message occurs before the 'no' user input log message
        log_messages = [record.message for record in caplog.records]
        invalid_choice_index = next(
            (
                i
                for i, msg in enumerate(log_messages)
                if "Invalid choice. Please enter " in msg
            ),
            None,
        )
        no_index = next(
            (i for i, msg in enumerate(log_messages) if "please check" in msg.lower()),
            None,
        )
        assert (
            invalid_choice_index is not None
            and no_index is not None
            and invalid_choice_index < no_index
        ), (
            "Expected 'Invalid choice. "
            "Please enter ' message before "
            "'no' input or 'please check' message"
        )

    def test_no_annuity_partial_all_cost_params_empty_raises_error(self):
        """
        Tests that a value error is raised when there is no annuity and not all of the cost
        parameters are defined.
        """
        # copy scenario csv file to the package path
        self.copy_file_to_package_path("no_annuity_partial_all_cost_params_empty.csv")
        # check if calling the pre_processing function with the given scenario raises a value error
        with pytest.raises(ValueError):
            pre_processing(self.pre_p_dir, wacc=1)

    def test_no_annuity_all_cost_params_defined_creates_annuity_param(self):
        """
        Tests that a new annuity parameter ('capacity_cost') is created when it is not included
        and all other cost parameters are defined.
        """
        # copy scenario csv file to the package path
        f_name = "no_annuity_all_cost_params_defined.csv"
        self.copy_file_to_package_path(f_name)
        # call the pre_processing function with wacc=1
        wacc = 1
        pre_processing(self.pre_p_dir, wacc=wacc)
        # check that the 'capacity_cost' parameter is added to the scenario csv file
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
        print(df.columns)
        assert "capacity_cost" in df.columns

    def test_no_annuity_all_cost_params_defined_calculates_annuity(self):
        """
        Tests that the annuity is calculated and added when it is not included and all other
        cost parameters are defined.
        """
        # copy scenario csv file to the package path
        f_name = "no_annuity_all_cost_params_defined.csv"
        self.copy_file_to_package_path(f_name)
        # call the pre_processing function with wacc=1
        wacc = 1
        pre_processing(self.pre_p_dir, wacc=wacc)
        # check that the actual value matches the expected value
        expected_value = calculate_annuity(
            capex=975000, opex_fix=11625, lifetime=20, wacc=wacc
        )
        df = pd.read_csv(os.path.join(self.package_path, f_name), delimiter=";")
        actual_value = df["capacity_cost"].iloc[0]
        assert actual_value == expected_value

    def test_no_annuity_all_cost_params_defined_logs_message(self, caplog):
        """
        Tests that an info message is logged when the annuity is not included and all other cost
        parameters are defined.
        """
        # copy scenario csv file to the package path
        f_name = "no_annuity_all_cost_params_defined.csv"
        shutil.copy(
            os.path.join(self.test_inputs_pre_p, f_name),
            os.path.join(self.package_path, f_name),
        )
        # check if the info message is logged when the pre_processing function is called
        pre_processing(self.pre_p_dir, wacc=1)
        assert any(
            record.levelname == "INFO"
            and "has been calculated and updated for" in record.message
            for record in caplog.records
        )

    def test_no_annuity_no_cost_params_logs_message(self, caplog):
        """
        Tests that an info message is logged when neither the annuity or any other cost parameters
        are defined.
        """
        # copy scenario csv file to the package path
        f_name = "no_annuity_no_cost_params.csv"
        shutil.copy(
            os.path.join(self.test_inputs_pre_p, f_name),
            os.path.join(self.package_path, f_name),
        )
        pre_processing(self.pre_p_dir, wacc=1)
        # check if the info message is logged when the pre_processing function is called
        assert any(
            record.levelname == "INFO" and "does not contain" in record.message
            for record in caplog.records
        )


class TestPreprocessingCustomAttributes:
    """
    This class contains tests for the pre-processing custom attributes function.
    """

    # store the relevant paths
    test_inputs_pre_p = os.path.join(
        "tests", "test_inputs_pre_processing_custom_attributes"
    )
    pre_p_dir = os.path.join(
        "tests", "test_inputs_pre_processing_custom_attributes", "pre_processing"
    )
    _package_path = os.path.join("data", "elements")
    package_path = _package_path

    def setup_method(self):
        """
        Sets up the testing environment before each test method is executed.
        """
        # removes the 'pre_processing' directory if it exists, ignoring errors
        if os.path.exists(self.pre_p_dir):
            shutil.rmtree(self.pre_p_dir, ignore_errors=True)
        # creates the 'pre_processing' directory
        os.mkdir(self.pre_p_dir)
        # creates the 'data' directory within the 'pre_processing' directory
        os.mkdir(os.path.join(self.pre_p_dir, "data"))
        # creates the 'elements' directory within the 'data' directory
        os.mkdir(os.path.join(self.pre_p_dir, self._package_path))
        # sets the package path for the test environment
        self.package_path = os.path.join(self.pre_p_dir, self._package_path)

    def copy_files_to_package_path(self, f_name, dp_name):
        """
        Copies the scenario CSV file and the datapackage JSON file to the package path
        :param f_name: scenario CSV filename
        :param dp_name: datapackage JSON filename
        """
        # copy scenario csv file and datapackage json file to the package path
        shutil.copy(
            os.path.join(self.test_inputs_pre_p, f_name),
            os.path.join(self.package_path, f_name),
        )
        shutil.copy(
            os.path.join(self.test_inputs_pre_p, dp_name),
            os.path.join(self.pre_p_dir, "datapackage.json"),
        )

    def teardown_method(self):
        """
        Cleans up the testing environment after each test method is executed.
        """
        # removes the 'pre_processing' directory if it exists, ignoring errors
        if os.path.exists(self.pre_p_dir):
            shutil.rmtree(self.pre_p_dir, ignore_errors=True)
        # resets the package path to its original value
        self.package_path = self._package_path

    def test_output_params_added_to_csv_with_cust_attr_in_csv_and_list(self):
        """
        Tests that the output parameters column has been added to the csv file if the csv
        file contains custom attributes and they have been defined as list by user.
        """
        # copy scenario csv file and datapackage json file to the package path
        f_name = "cust_attr.csv"
        self.copy_files_to_package_path(f_name, "dp_no_output_params.json")
        # call the pre_processing function with wacc = 1 and custom_attributes list defined
        wacc = 1
        pre_processing(
            self.pre_p_dir,
            wacc=wacc,
            custom_attributes=[
                "emission_factor",
                "renewable_factor",
                "land_requirement",
            ],
        )
        # read the updated csv file
        updated_df = pd.read_csv(os.path.join(self.package_path, f_name), sep=";")
        print(updated_df.columns)
        # assert that 'output_parameters' column is in the updated dataframe
        assert "output_parameters" in updated_df.columns, (
            "'output_parameters' column is not present " "in the updated dataframe"
        )

    def test_output_params_not_added_to_csv_with_cust_attr_in_csv_and_not_list(self):
        """
        Tests that the output parameters column has not been added to the csv file if the csv
        file contains custom attributes and they have not been defined as list by user.
        """
        # copy scenario csv file and datapackage json file to the package path
        f_name = "cust_attr.csv"
        self.copy_files_to_package_path(f_name, "dp_no_output_params.json")
        # read the original dataframe before pre-processing
        original_df = pd.read_csv(os.path.join(self.package_path, f_name), sep=";")
        # call the pre_processing function with wacc = 1 and custom_attributes = none (default)
        wacc = 1
        pre_processing(self.pre_p_dir, wacc=wacc)
        # read the updated csv file after pre-processing
        updated_df = pd.read_csv(os.path.join(self.package_path, f_name), sep=";")
        # assert that 'output_parameters' column is not in the updated dataframe if it wasn't present initially
        if "output_parameters" not in original_df.columns:
            assert (
                "output_parameters" not in updated_df.columns
            ), "'output_parameters' has been added to the updated dataframe when it should not be"

    def test_output_params_added_to_json_with_cust_attr_in_csv_and_list(self):
        """
        Tests that the output parameters field has been added to the datapackage.json file if the
        csv file contains custom attributes and they have been defined as list by user.
        """
        # copy scenario csv file and datapackage json file to the package path
        self.copy_files_to_package_path("cust_attr.csv", "dp_no_output_params.json")
        # call the pre_processing function with wacc = 1 and custom_attributes list defined
        wacc = 1
        pre_processing(
            self.pre_p_dir,
            wacc=wacc,
            custom_attributes=[
                "emission_factor",
                "renewable_factor",
                "land_requirement",
            ],
        )
        # read the datapackage.json file after pre-processing
        with open(os.path.join(self.pre_p_dir, "datapackage.json"), "r") as f:
            updated_datapackage = json.load(f)
        # get the resource from the datapackage.json file
        resource = updated_datapackage.get("resources", [None])[0]
        # check if the resource was found
        assert (
            resource is not None
        ), "No resource found in the updated datapackage.json file"
        # find the appropriate field within the resource's schema
        fields = resource.get("schema", {}).get("fields", [])
        output_parameters_field = None
        for field in fields:
            if field.get("name") == "output_parameters":
                output_parameters_field = field
                break
        # assert that the 'output_parameters' field has been added to the datapackage.json file
        assert output_parameters_field is not None, (
            "output_parameters field is not present in the updated "
            "datapackage.json file"
        )

    def test_output_params_not_added_to_json_with_cust_attr_in_csv_and_not_list(self):
        """
        Tests that the output parameters field has not been added to the datapackage.json file if the
        csv file contains custom attributes and they have not been defined as list by user.
        """
        # copy scenario csv file and datapackage json file to the package path
        self.copy_files_to_package_path("cust_attr.csv", "dp_no_output_params.json")
        # call the pre_processing function with wacc = 1 and custom_attributes = none (default)
        wacc = 1
        pre_processing(self.pre_p_dir, wacc=wacc)
        # read the datapackage.json file after pre-processing
        with open(os.path.join(self.pre_p_dir, "datapackage.json"), "r") as f:
            updated_datapackage = json.load(f)
        with open(
            os.path.join(self.test_inputs_pre_p, "dp_no_output_params.json"), "r"
        ) as f:
            original_datapackage = json.load(f)
        assert updated_datapackage == original_datapackage, (
            "The datapackage.json file has been updated " "when it shouldn't be"
        )

    def test_output_params_not_added_to_twice_when_already_exists(self):
        """
        Tests that the output parameters field has not been added again to the datapackage.json file
        when it already exists
        """
        # copy scenario csv file and datapackage json file to the package path
        self.copy_files_to_package_path("cust_attr.csv", "dp_output_params.json")
        # call the pre_processing function with wacc = 1 and custom_attributes list defined
        wacc = 1
        pre_processing(
            self.pre_p_dir,
            wacc=wacc,
            custom_attributes=[
                "emission_factor",
                "renewable_factor",
                "land_requirement",
            ],
        )
        # read the datapackage.json file after pre-processing
        with open(os.path.join(self.pre_p_dir, "datapackage.json"), "r") as f:
            updated_datapackage = json.load(f)
        # get the resource from the datapackage.json file
        resource = updated_datapackage.get("resources", [None])[0]
        # check if the resource was found
        assert (
            resource is not None
        ), "No resource found in the updated datapackage.json file"
        # find the appropriate field within the resource's schema
        fields = resource.get("schema", {}).get("fields", [])
        output_parameters_count = sum(
            1 for field in fields if field.get("name") == "output_parameters"
        )
        # assert that the 'output_parameters' field exists only once
        assert (
            output_parameters_count == 1
        ), "The 'output_parameters' field exists more than once in the schema"
