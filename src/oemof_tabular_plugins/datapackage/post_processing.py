import pandas as pd
import oemof.solph as solph


def infer_busses_carrier(energy_system):
    """Loop through the nodes of an energy system and infer the carrier of busses from them

    Parameters
    ----------
    energy_system: oemof.solph.EnergySystem instance

    Returns
    -------
    dict mapping the busses labels to their carrier

    """

    busses_carrier = {}

    for node in energy_system.nodes:
        if hasattr(node, "carrier"):
            for attribute in ("bus", "from_bus"):
                if hasattr(node, attribute):

                    bus_label = getattr(node, attribute).label
                    if bus_label in busses_carrier:
                        if busses_carrier[bus_label] != node.carrier:
                            raise ValueError(
                                f"Two different carriers ({busses_carrier[bus_label]}, {node.carrier}) are associated to the same bus '{bus_label}'"
                            )
                    else:
                        busses_carrier[bus_label] = node.carrier

    busses = [node.label for node in energy_system.nodes if isinstance(node, solph.Bus)]

    for bus_label in busses:
        if bus_label not in busses_carrier:
            raise ValueError(
                f"Bus '{bus_label}' is missing from the busses carrier dict inferred from the EnergySystem instance"
            )

    return busses_carrier


def infer_asset_types(energy_system):
    """Loop through the nodes of an energy system and infer their types

    Parameters
    ----------
    energy_system: oemof.solph.EnergySystem instance

    Returns
    -------
    a dict mapping the asset (nodes which are not busses) labels to their type

    """
    asset_types = {}
    for node in energy_system.nodes:
        if isinstance(node, solph.Bus) is False:
            asset_types[node.label] = node.type
    return asset_types


def construct_multi_index_levels(flow_tuple, busses_info, assets_info=None):
    """Infer the index levels of the multi index dataframe sequence tuple and extra optional information

    Parameters
    ----------
    flow_tuple: tuple of bus label and asset label
        (A,B) means flow from A to B
    busses_info: either a list or a dict
        if not a list of busses labels, then a dict where busses labels are keys mapped to the bus carrier
    assets_info: dict
        mapping of asset labels to their type

    Returns
    -------
    a tuple with (bus label, direction of flow relative to asset, asset label, bus carrier (optional), asset type (optional)) direction is either 'in' or 'out'.

    The minimal tuple (b_elec, "in", demand) would read the flow goes from bus 'b_elec' '"in"' asset 'demand'

    """
    if isinstance(busses_info, dict):
        busses_labels = [bn for bn in busses_info]
    elif isinstance(busses_info, list):
        busses_labels = busses_info

    # infer which of the 2 nodes composing the flow is the bus
    bus_label = set(busses_labels).intersection(set(flow_tuple))
    if len(bus_label) == 1:
        bus_label = bus_label.pop()
    else:
        raise ValueError("Flow tuple does not contain only one bus node")
    # get position of bus node in the flow tuple
    idx_bus = flow_tuple.index(bus_label)
    answer = None

    # determine whether the flow goes from bus to asset or reverse
    if idx_bus == 0:
        # going from bus to asset, so the flow goes in to the asset
        asset_label = flow_tuple[1]
        answer = (bus_label, "in", asset_label)

    elif idx_bus == 1:
        asset_label = flow_tuple[0]
        # going from asset to bus, so the flow goes out of the asset
        answer = (bus_label, "out", asset_label)

    # add information of the bus carrier, if provided
    if isinstance(busses_info, dict):
        answer = answer + (busses_info[bus_label],)
    # add information of the asset type, if provided
    if assets_info is not None:
        answer = answer + (assets_info[asset_label],)
    return answer


def construct_dataframe_from_results(energy_system, bus_carrier=True, asset_type=True):
    """

    Parameters
    ----------
    energy_system: oemof.solph.EnergySystem instance
    bus_carrier: bool (opt)
        If set to true, the multi-index of the DataFrame will have a level about bus carrier
    asset_type: bool (opt)
        If set to true, the multi-index of the DataFrame will have a level about the asset type


    Returns
    -------
    Dataframe with oemof result sequences's timestamps as columns as well as investment and a multi-index built automatically, see construct_multi_index_levels for more information on the multi-index
    """
    mi_levels = [
        "bus",
        "direction",
        "asset",
    ]

    busses_info = infer_busses_carrier(energy_system)
    if bus_carrier is False:
        busses_info = list(busses_info.keys())
    else:
        mi_levels.append("carrier")

    if asset_type is True:
        assets_info = infer_asset_types(energy_system)
        mi_levels.append("facade_type")
    else:
        assets_info = None

    results = energy_system.results

    if isinstance(results, dict):
        ts = []
        investments = []
        flows = []
        for x, res in solph.views.convert_keys_to_strings(results).items():
            if x[1] != "None":
                col_name = res["sequences"].columns[0]
                ts.append(
                    res["sequences"].rename(
                        columns={col_name: x, "variable_name": "timesteps"}
                    )
                )
                flows.append(
                    construct_multi_index_levels(
                        x, busses_info=busses_info, assets_info=assets_info
                    )
                )
                invest = None if res["scalars"].empty is True else res["scalars"].invest

                investments.append(invest)
        ts_df = pd.concat(ts, axis=1, join="inner")
        mindex = pd.MultiIndex.from_tuples(flows, names=mi_levels)

        df = pd.DataFrame(
            data=ts_df.T.to_dict(orient="split")["data"],
            index=mindex,
            columns=ts_df.index,
        )

        df["investments"] = investments
        df.sort_index(inplace=True)

    return df
