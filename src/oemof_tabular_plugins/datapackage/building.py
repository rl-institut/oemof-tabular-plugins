import os
import warnings
import pandas as pd
from oemof.tabular.datapackage import building
from datapackage import Package

# TODO when oemof.tabular is updated with inferred metadata PR, use
# from oemof.tabular.config import config
import oemof_tabular_plugins.datapackage.config as config


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
    data = pd.DataFrame.from_records(r.read(keyed=True))
    # TODO not sure this should be set here
    r.descriptor["schema"]["primaryKey"] = "name"
    if "foreignKeys" not in r.descriptor["schema"]:
        r.descriptor["schema"]["foreignKeys"] = []

    for field in r.schema.fields:
        if field.type == "string":
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
                if potential_fk in busses:
                    fk = {
                        "fields": field.name,
                        "reference": {"resource": "bus", "fields": "name"},
                    }
                    if fk not in r.descriptor["schema"]["foreignKeys"]:
                        r.descriptor["schema"]["foreignKeys"].append(fk)

    r.commit()
    return r


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
                    facade_type.validate_datapackage(r)

            p.remove_resource(r.name)
            p.add_resource(r.descriptor)


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
        for field in resource.schema.fields:
            if field.name in config.SPECIAL_FIELD_NAMES:
                fk_descriptor = config.SPECIAL_FIELD_NAMES[field.name]
                if fk_descriptor in foreign_keys:
                    if resource.name not in foreign_keys[fk_descriptor]:
                        foreign_keys[fk_descriptor].append(resource.name)
                else:
                    foreign_keys[fk_descriptor] = [resource.name]

    for r in p0.resources:
        if os.sep + "elements" + os.sep in r.descriptor["path"]:
            infer_resource_basic_foreign_keys(r)
    # this function saves the metadata of the package in json format
    building.infer_metadata(
        package_name=package_name,
        path=path,
        foreign_keys=foreign_keys,
        metadata_filename=metadata_filename,
    )

    # reload the package from the saved json file
    p = Package(os.path.join(path, metadata_filename))
    infer_package_foreign_keys(p, typemap=typemap)
    p.descriptor["resources"].sort(key=lambda x: (x["path"], x["name"]))
    p.commit()
    p.save(os.path.join(path, metadata_filename))
