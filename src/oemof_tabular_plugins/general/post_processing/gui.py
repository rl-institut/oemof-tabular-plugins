import logging

import numpy as np
import os
from datapackage import Package
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta
from oemof import solph
from oemof.tools.economics import annuity

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from oemof.visio import ESGraphRenderer

    ES_GRAPH = True
except ModuleNotFoundError:
    ES_GRAPH = False
z_version = 1


import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

RESULTS_COLUMN_NAMES = [
    "annuity",
    "annual_costs",
    "total_flow",
    "capacity",
    "cash_flow",  # AA: could be named fuel_expenditure_cost
    "total_opex_costs",
    "first_investment",
]

service__item_style = {
    "border-style": "solid",
    "border-width": "3px",
    "padding": "1rem",
    "margin": "1rem",
    "border-radius": "5px",
}

table__item_style = {
    "kpis": {
        "width": "500px",
        "border-style": "solid",
        "border-width": "3px",
        "padding": "1rem",
        "margin": "1rem",
        "border-radius": "5px",
    },
    "capacities": {
        "width": "800px",
        "border-style": "solid",
        "border-width": "3px",
        "padding": "1rem",
        "margin": "1rem",
        "border-radius": "5px",
    },
}

container_style = {
    "display": "flex",
    "flex-direction": "row",
    "flex-wrap": "wrap",
    "justify-content": "flex-start",
}


##########################################################################
# Initialize the energy system and calculate necessary parameters
##########################################################################


def encode_image_file(img_path):
    """Encode image files to load them in the dash layout under img html tag

    Parameters
    ----------
    img_path: str
        path to the image file

    Returns
    -------
    encoded_img: bytes
        encoded bytes of the image file

    """

    try:
        with open(img_path, "rb") as ifs:
            encoded_img = base64.b64encode(ifs.read())
    except FileNotFoundError:
        encoded_img = base64.b64encode(bytes())
    return encoded_img


def sankey(results, display_name, date_time_index=None, ts=None):
    """
    Return a dict for a Plotly Sankey diagram, using df_results (MultiIndex).
    Optionally, select a single timestep `ts` for the flow values.
    """
    labels = []
    sources = []
    targets = []
    values = []

    # Extract all buses from df_results
    busses = results.index.get_level_values("bus").unique()

    for bus in busses:
        # Skip buses not in the results
        if bus not in results.index.get_level_values("bus"):
            logging.warning(f"Bus '{bus}' not found in results, skipping in Sankey.")
            continue

        bus_df = results.loc[bus]

        if bus_df is None or bus_df.empty:
            logging.warning(f"No flows found for bus '{bus}', skipping in Sankey.")
            continue

        for direction, asset, carrier, facade_type in bus_df.index:
            row = bus_df.loc[(direction, asset, carrier, facade_type)]

            # Determine source and target depending on flow direction with df_results semantics:
            # direction == "in"  → flow FROM bus TO asset
            # direction == "out" → flow FROM asset TO bus

            if direction == "in":
                source_label = display_name(bus)
                target_label = display_name(asset)
            elif direction == "out":
                source_label = display_name(asset)
                target_label = display_name(bus)
            else:
                continue

            # Add labels if not already present
            for lbl in (source_label, target_label):
                if lbl not in labels:
                    labels.append(lbl)

            # Get flow value
            if date_time_index is not None:
                # intersect datetime index with actual row columns
                ts_cols = [c for c in date_time_index if c in row.index]
            else:
                # fallback: pick all datetime-like columns
                ts_cols = [c for c in row.index if not pd.isna(pd.to_datetime(c, errors="coerce"))]

            if ts_cols:
                ts_series = row[ts_cols].fillna(0)
                if ts is not None:
                    flow_value = ts_series.iloc[ts] if ts < len(ts_series) else 0
                else:
                    flow_value = ts_series.sum()
            else:
                flow_value = 0

            sources.append(labels.index(source_label))
            targets.append(labels.index(target_label))
            values.append(flow_value)

    # Build the Sankey figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    hovertemplate="Node has total value %{value}<extra></extra>",
                    color="blue",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    hovertemplate=(
                        "Link from node %{source.label}<br />"
                        + "to node %{target.label}<br />has value %{value}<extra></extra>"
                    ),
                ),
            )
        ]
    )

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    return fig.to_dict()



def prepare_app(app, dp_path, results, tables, services, units=None):
    """ """
    p0 = Package(dp_path)

    # Dynamic label mapping to use verbose names (if available)
    label_map = {}

    for resource_name in p0.resource_names:
        try:
            df = pd.DataFrame.from_records(p0.get_resource(resource_name).read(keyed=True))

            if "name" in df.columns and "verbose_name" in df.columns:
                label_map.update(
                    {
                        row["name"]: row["verbose_name"]
                        for _, row in df.iterrows()
                        if pd.notna(row["verbose_name"])
                    }
                )

        except Exception:
            pass

    def display_name(name):
        return label_map.get(name, name)

    # Derive datetime index from results
    time_cols = [
        c for c in results.columns
        if not pd.isna(pd.to_datetime(c, errors="coerce"))
    ]

    date_time_index = pd.to_datetime(time_cols)

    # List for bus figures
    bus_figures = []

    # Only plot busses that a) have the parameter "plot" == True and b) are in the results
    bus_data = pd.DataFrame.from_records(p0.get_resource("bus").read(keyed=True))
    available_busses = set(results.index.get_level_values("bus"))

    busses = bus_data.loc[
        bus_data["plot"].fillna(False)
        & bus_data["name"].isin(available_busses),
        "name",
    ].tolist()

    import pdb; pdb.set_trace()

    for bus in busses:
        fig = go.Figure(layout=dict(title=f"{display_name(bus)} bus node"))

        bus_df = results.loc[bus]
        for (direction, asset, carrier, facade_type), row in bus_df.iterrows():
            flow_values = row[date_time_index].values

            sign = 1 if direction == "out" else -1

            fig.add_trace(
                go.Scatter(
                    x=date_time_index,
                    y=flow_values * sign,
                    name=display_name(asset),
                )
            )

        # else:
        #     capacity_battery = asset_results.capacity.battery
        #     if capacity_battery != 0:
        #         soc_battery = solph.views.node(results, node=bus)["sequences"][
        #                           (("battery", "None"), "storage_content")] / capacity_battery
        #     else:
        #         soc_battery = solph.views.node(results, node=bus)["sequences"][
        #             (("battery", "None"), "storage_content")]
        #
        #     fig = go.Figure(layout=dict(title=f"{bus} node"))
        #
        #     fig.add_trace(
        #         go.Scatter(
        #             x=soc_battery.index, y=soc_battery.values, name="soc battery"
        #         )
        #     )

        bus_figures.append(fig)

    tables_figure = []

    for table in tables:
        df = tables[table]

        if "unit" not in df.columns:
            df.reset_index(inplace=True)

            def set_value(row_number, assigned_value):
                return assigned_value.get(row_number, None)

            df["unit"] = df[df.columns[0]].apply(set_value, args=(units,))

        if "Component name" in df.columns:
            df["Component name"] = df["Component name"].apply(display_name)
        tables_figure.append(
            html.Div(
                style=table__item_style[table],
                children=[
                    html.H4(table),
                    dash_table.DataTable(
                        data=df.round(2).to_dict("records"),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_cell_conditional=[
                            {"if": {"column_id": "kpi"}, "textAlign": "center"},
                            {
                                "if": {"column_id": "Component name"},
                                "textAlign": "center",
                            },
                        ],
                    ),
                ],
            )
        )

    services_figure = []
    for service in services:
        df = services[service]
        carrier = df.carrier.unique()[0]
        unit = units.get(carrier, "UNIT NOT FOUND")

        is_crop = False

        if "crop" in df.facade_type.values:
            is_crop = True

        df = df.rename(
            columns={"asset": "Component name", "aggregated_flow": unit},
        )

        # Populate the subtables of the service
        production = df.loc[df.direction == "out"].copy()

        if is_crop is False:
            usage = df.loc[(df.direction == "in") & (df.facade_type != "excess")].copy()
            excess = df.loc[df.facade_type == "excess"].copy()
            tables = [production, usage, excess]
        else:
            usage = df.loc[df.direction == "in"].copy()
            tables = [production, usage]

        for table in tables:
            if table.empty is False:
                table.drop(
                    columns=["carrier", "direction", "facade_type"], inplace=True
                )
                table.loc[:, "Percentage"] = 100 * table[unit] / table[unit].sum()
                # Add a line with "total" if there is more than one component
                if len(table) > 1:
                    summary_line = table.iloc[:, 1:].sum()
                    summary_line["Component name"] = "Total"
                    table.loc[-1] = summary_line

        table_headers = ["Production", "Usage"]
        if is_crop is False:
            if excess.empty is False:
                if excess[unit].sum() > 0:
                    table_headers.append("Excess")

        if "Component name" in df.columns:
            df["Component name"] = df["Component name"].apply(display_name)
        services_figure.append(
            html.Div(
                id=f"{service}-service-div",
                className="service--item",
                style=service__item_style,
                children=[
                    html.H4(service.replace("-", " ").capitalize()),
                    html.Div(
                        children=[
                            html.Div(
                                [
                                    html.H5(table_hdr),
                                    dash_table.DataTable(
                                        data=table.round(2).to_dict("records"),
                                        columns=[
                                            {"name": i, "id": i} for i in table.columns
                                        ],
                                        style_cell_conditional=[
                                            {
                                                "if": {"column_id": i},
                                                "width": f"{100/len(table.columns)}%",
                                            }
                                            for i in table.columns
                                        ]
                                        + [
                                            {
                                                "if": {"column_id": "Component name"},
                                                "textAlign": "center",
                                            }
                                        ],
                                    ),
                                ]
                            )
                            for table_hdr, table in zip(table_headers, tables)
                        ],
                    ),
                ],
            )
        )

    # loading external resources
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

    app.layout = html.Div(
        children=[
            html.H2("Scalar results"),
            html.H3("KPIS"),
            html.Div(
                className="table--container",
                style=container_style,
                children=tables_figure,
            ),
            html.H3("Services"),
            # dcc.Dropdown(
            #     options=[s for s in services],
            #     value=[s for s in services],
            #     id="service_select",
            #     multi=True
            # ),
            dcc.Checklist(
                id="service_select",
                options=[s for s in services],
                value=[],  # s for s in services],
                inline=True,
            ),
            html.Div(
                services_figure,
                className="service--container",
                style=container_style,
            ),
            html.H2("Dynamic results"),
            html.P(
                children=[
                    "You can adjust the slider to get the energy flow at a single timestep, "
                    "or look for a specific timestep in the dropdown menu below ",
                    html.Span(
                        "Note if you change the slider "
                        "it will show the value in the dropdown menu, but it you change the dropdown menu directly "
                        "it will not update the slider)"
                    ),
                ]
            ),
            dcc.Slider(
                id="ts_slice_slider",
                value=1,
                min=0,
                max=len(date_time_index),
                # marks={k: v for k, v in enumerate(date_time_index)},
            ),
            dcc.Dropdown(
                id="ts_slice_select",
                options={k: v for k, v in enumerate(date_time_index)},
                value=None,
            ),
            dcc.Graph(id="sankey", figure=sankey(results, display_name, date_time_index)),
        ]
        + [
            dcc.Graph(
                id=f"{bus}-id",
                figure=fig,
            )
            for bus, fig in zip(busses, bus_figures)
        ]
        + [dcc.Graph(id="sankey_aggregate", figure=sankey(results, display_name, date_time_index))]
        # + [
        #     html.H4(["Energy system"]),
        #     html.Img(
        #         src="data:image/png;base64,{}".format(energy_system_graph.decode()),
        #         alt="Energy System Graph, if you do not see this image it is because pygraphviz is not installed. "
        #             "If you are a windows user it might be complicated to install pygraphviz.",
        #         style={"maxWidth": "100%"},
        #     ),
        # ]
    )

    @app.callback(
        # The value of these components of the layout will be changed by this callback
        [
            Output(component_id="sankey", component_property="figure"),
        ]
        + [
            Output(component_id=f"{bus}-id", component_property="figure")
            for bus in busses
        ],
        # Triggers the callback when the value of one of these components of the layout is changed
        Input(component_id="ts_slice_select", component_property="value"),
    )
    def update_figures(ts):
        if ts is None:
            ts = "0"
        ts = int(ts)

        bus_figures = []

        for bus in busses:
            fig = go.Figure(layout=dict(title=f"{display_name(bus)} bus node"))
            max_y = 0

            # Skip if bus not in results
            if bus not in results.index.get_level_values("bus"):
                logging.warning(f"Bus '{bus}' not found in results.")
                bus_figures.append(fig)
                continue

            bus_df = results.loc[bus]
            if bus_df is None or bus_df.empty:
                logging.warning(f"No flows found for bus '{bus}'.")
                bus_figures.append(fig)
                continue

            for direction, asset, carrier, facade_type in bus_df.index:
                row = bus_df.loc[(direction, asset, carrier, facade_type)]

                # Determine sign for plotting
                negative_sign = -1 if direction == "in" else 1
                asset_name = display_name(asset)
                if asset == "battery":
                    asset_name += " discharge" if direction == "out" else " charge"

                # Safe time series extraction
                ts_cols = [c for c in date_time_index if c in row.index]
                if not ts_cols:
                    continue
                y_values = row[ts_cols].fillna(0).values

                fig.add_trace(
                    go.Scatter(
                        x=ts_cols,
                        y=y_values * negative_sign,
                        name=asset_name,
                        stackgroup="negative_sign" if negative_sign < 0 else "positive_sign",
                    )
                )

                if y_values.size > 0:
                    max_y = max(max_y, abs(y_values).max())

            # Vertical line at current timestep
            if ts < len(date_time_index):
                fig.add_trace(
                    go.Scatter(
                        x=[date_time_index[ts], date_time_index[ts]],
                        y=[0, max_y],
                        name="current timestep",
                        line_color="black",
                    )
                )

            bus_figures.append(fig)

        return [sankey(results, display_name, date_time_index, ts)] + bus_figures

    @app.callback(
        # The value of these components of the layout will be changed by this callback
        Output(component_id="ts_slice_select", component_property="value"),
        # Triggers the callback when the value of one of these components of the layout is changed
        Input(component_id="ts_slice_slider", component_property="value"),
    )
    def change_ts_value(val):
        return val

    @app.callback(
        # The value of these components of the layout will be changed by this callback
        [
            Output(component_id=f"{s}-service-div", component_property="style")
            for s in services
        ],
        # Triggers the callback when the value of one of these components of the layout is changed
        Input(component_id="service_select", component_property="value"),
    )
    def change_visibility_value(val):
        answer = [
            (
                service__item_style | {"display": "block"}
                if s in val
                else service__item_style | {"display": "none"}
            )
            for s in services
        ]
        return answer

    return app

    # import ipdb;ipdb.set_trace()
