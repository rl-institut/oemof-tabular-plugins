import os
import warnings
import logging
import pandas as pd
from tableschema.exceptions import CastError
from oemof.tabular.datapackage import building
from datapackage import Package

# TODO when oemof.tabular is updated with inferred metadata PR, use
# from oemof.tabular.config import config
import oemof_tabular_plugins.datapackage.config as config

PROFILE_FIELDS = ["profile"]

STR_FIELDS = ["name", "carrier", "tech", "type", "region", "crop_type"]


def map_sequence_profiles_to_resource_name(p, excluded_profiles=("timeindex",)):
    """Look in every resource which is a sequence and map each of its fields to itself

    Within this process the unicity of the field names will be checked, with the exception of the field "timeindex"

    """

    def check_sequences_labels_unicity(labels, new_labels):
        intersect = set(labels).intersection(new_labels)
        if len(intersect) == 1:
            intersect = intersect.pop()
            if not intersect == "timeindex":
                answer = [intersect]
            else:
                answer = []
        else:
            answer = list(intersect)

        if answer:
            warnings.warn(
                f"The labels of the profiles are not unique across all files within 'sequences' folder: '{','.join(intersect)}' used more than once"
            )
        return answer

    sequences = {}
    sequence_labels = []
    duplicated_labels = []
    for r in p.resources:
        if "/sequences/" in r.descriptor["path"]:
            field_labels = [
                f.name for f in r.schema.fields if f.name not in excluded_profiles
            ]
            sequences[r.descriptor["name"]] = field_labels
            duplicated_labels += check_sequences_labels_unicity(
                sequence_labels, field_labels
            )
            sequence_labels += field_labels

    if duplicated_labels:
        # write an error message here
        pass
    # map each profile to its resource name
    sequences_mapping = {
        value: key for (key, values) in sequences.items() for value in values
    }
    return sequences_mapping


def infer_resource_foreign_keys(resource, sequences_profiles_to_resource, busses):
    """Find out the foreign keys within a resource fields

    Look through all field of a resource which are of type 'string' if any of their values are matching a profile header in any of the sequences resources


    Parameters
    ----------
    resource: a :datapackage.Resource: instance
    sequences_profiles_to_resource: the mapping of sequence profile headers to their resource name

    Returns
    -------
    The :datapackage.Resource: instance with updated "foreignKeys" field

    """
    r = resource
    try:
        data = pd.DataFrame.from_records(r.read(keyed=True))
    except CastError:
        raise ValueError(
            f"Error while parsing the resource '{r.descriptor.get('path', r.name)}'. Check that all lines in the file have same number of records. Sometimes it might also a ',' instead of ';' as separator or vice versa."
        )
    # TODO not sure this should be set here
    r.descriptor["schema"]["primaryKey"] = "name"
    if "foreignKeys" not in r.descriptor["schema"]:
        r.descriptor["schema"]["foreignKeys"] = []

    for field in r.schema.fields:
        if field.type == "string" and field.name not in STR_FIELDS:
            for potential_fk in data[field.name].dropna().unique():
                if potential_fk in sequences_profiles_to_resource:
                    # this is actually a wrong format and should be with a "fields" field under the "reference" fields

                    fk = {
                        "fields": field.name,
                        "reference": {
                            "resource": sequences_profiles_to_resource[potential_fk],
                        },
                    }

                    if fk not in r.descriptor["schema"]["foreignKeys"]:
                        r.descriptor["schema"]["foreignKeys"].append(fk)
                elif potential_fk in busses:
                    fk = {
                        "fields": field.name,
                        "reference": {"resource": "bus", "fields": "name"},
                    }
                    if fk not in r.descriptor["schema"]["foreignKeys"]:
                        r.descriptor["schema"]["foreignKeys"].append(fk)
                else:
                    # check for specific fields which are meant to link to profile
                    possible_field_values = [
                        f"'{seq}'" for seq in sequences_profiles_to_resource
                    ]
                    logging.error(
                        f"The value '{potential_fk}' of the field '{field.name}' of the resource '{r.name}' does not match the headers of any sequences. "
                        "If this field is not meant to be a foreign key, you can safely ignore this error :) "
                        f"If this field is meant to be a foreign key to a sequence, then possible values are: {','.join(possible_field_values)}"
                    )

    r.commit()
    return r


def check_profiles(package):
    """Check that values of foreign keys to resources in data/sequences have a match in target sequence headers
    Parameters
    ----------
    package: datapackage instance
    Returns
    -------
    None, raises an error if a value assigned under the foreign key field does not match target sequence headers

    """
    for r in package.resources:
        fkeys = r.descriptor["schema"].get("foreignKeys", [])
        if fkeys:

            data = pd.DataFrame.from_records(r.read(keyed=True))
            for foreign_key in fkeys:
                fk_field = foreign_key["fields"]
                fk_target = foreign_key["reference"]["resource"]
                if fk_target != "bus":
                    for fk_value in data[fk_field].dropna().unique():
                        sequence_descriptor = package.get_resource(fk_target).descriptor
                        sequence_headers = [
                            f"{f['name']}"
                            for f in sequence_descriptor["schema"].get("fields", [])
                        ]
                        if fk_value not in sequence_headers:
                            raise ValueError(
                                f"The value {fk_value} of the field {fk_field} within the resource {r.name} does not match the headers of its resource within '{sequence_descriptor['path']}'\n"
                                f"possible values for the field {fk_field} are: {', '.join(sequence_headers)}"
                            )


def infer_package_foreign_keys(package, typemap=None):
    """Infer the foreign_keys from data/elements and data/sequences and update meta data

    Parameters
    ----------
    package: scenario datapackage
    typemap: facade typemap

    """
    if typemap is None:
        typemap = {}

    p = package
    sequences_profiles_to_resource = map_sequence_profiles_to_resource_name(p)

    bus_data = pd.DataFrame.from_records(p.get_resource("bus").read(keyed=True))

    for r in p.resources:
        # it should be from the folder "elements" and the resource should not be bus.csv
        if (
            "/elements/" in r.descriptor["path"]
            or os.sep + "elements" + os.sep in r.descriptor["path"]
        ) and r.name != "bus":
            r = infer_resource_foreign_keys(
                r, sequences_profiles_to_resource, busses=bus_data.name.to_list()
            )

            if r.name in typemap:
                facade_type = typemap[r.name]
                # test if facade_type has the method 'validate_datapackage'
                if hasattr(facade_type, "validate_datapackage"):
                    # apply the method if it exists
                    r = facade_type.validate_datapackage(r)

            p.remove_resource(r.name)
            p.add_resource(r.descriptor)


def infer_busses_carrier(package, infer_from_component=True):
    """Loop through the nodes of an energy system and infer the carrier of busses from them

    Parameters
    ----------
    package: datapackage.Package instance
    infer_from_component: bool
        if True, the bus carrier mapping will be inferred from the components if not found in 'bus'
        resource of the package

    Returns
    -------
    dict mapping the busses labels to their carrier

    """

    bus_data = pd.DataFrame.from_records(package.get_resource("bus").read(keyed=True))
    bus_carrier = None
    if "carrier" in bus_data.columns:
        bus_carrier = {row[1]["name"]: row[1]["carrier"] for row in bus_data.iterrows()}

    if bus_carrier is None:
        busses_carrier = {}
        if infer_from_component is True:
            for node_type in package.resources:
                if node_type.name != "bus":
                    fields = [f.name for f in node_type.schema.fields]
                    if "carrier" in fields:
                        nodes = node_type.read(keyed=True)
                        for attribute in ("bus", "from_bus", "from_bus_0", "to_bus_1"):
                            for node in nodes:
                                if attribute in node and node["carrier"] != "":

                                    bus_label = node[attribute]
                                    if bus_label in busses_carrier:
                                        if busses_carrier[bus_label] != node["carrier"]:
                                            raise ValueError(
                                                f"Two different carriers ({busses_carrier[bus_label]}, {node['carrier']}) are associated to the same bus '{bus_label}'"
                                            )
                                    else:
                                        busses_carrier[bus_label] = node["carrier"]
        else:
            return
    else:
        logging.info(
            "Bus to carrier mapping found in 'elements/bus.csv' file of datapackage"
        )
        busses_carrier = bus_carrier

    if not busses_carrier:
        raise ValueError(
            "The bus-carrier mapping is empty, this is likely due to missing 'carrier' attributes in the csv files of the elements folder of the datapackage. The simpler way to fix this, is to add a 'carrier' column in the 'elements/bus.csv' file"
        )

    # Check that every bus has a carrier assigned to it
    busses = bus_data.name.to_list()

    for bus_label in busses:
        if bus_label not in busses_carrier:
            print("busses carriers", busses_carrier)
            raise ValueError(
                f"Bus '{bus_label}' is missing from the busses carrier dict inferred from the EnergySystem instance"
            )

    return busses_carrier


def infer_metadata_from_data(
    package_name="default-name",
    path=None,
    metadata_filename="datapackage.json",
    typemap=None,
):
    """

    Returns
    -------

    """

    # Infer the fields from the package data
    path = os.path.abspath(path)
    p0 = Package(base_path=path)
    p0.infer(os.path.join(path, "**" + os.sep + "*.csv"))
    p0.commit()
    p0.save(os.path.join(path, metadata_filename))

    foreign_keys = {}

    def infer_resource_basic_foreign_keys(resource):
        """insert resource foreign_key into a dict formatted for building.infer_metadata

        Compare the fields of a resource to a list of field names known to be foreign keys. If the field name is within the list, it is used to populate the dict 'foreign_keys'
        """

    for r in p0.resources:
        if os.sep + "elements" + os.sep in r.descriptor["path"]:
            infer_resource_basic_foreign_keys(r)
    # this function saves the metadata of the package in json format
    building.infer_metadata(
        package_name=package_name,
        path=path,
        metadata_filename=metadata_filename,
    )

    # reload the package from the saved json file
    p = Package(os.path.join(path, metadata_filename))
    infer_package_foreign_keys(p, typemap=typemap)
    p.descriptor["resources"].sort(key=lambda x: (x["path"], x["name"]))
    p.commit()
    p.save(os.path.join(path, metadata_filename))
    infer_busses_carrier(p)
    check_profiles(p)
