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
    from oemof_visio import ESGraphRenderer

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


def sankey(energy_system, ts=None):
    """Return a dict to a plotly sankey diagram"""
    busses = []

    labels = []
    sources = []
    targets = []
    values = []

    results = energy_system.results

    # draw a node for each of the network's component. The shape depends on the component's type
    for nd in energy_system.nodes:
        if isinstance(nd, solph.Bus):

            # keep the bus reference for drawing edges later
            bus = nd
            busses.append(bus)

            bus_label = bus.label

            labels.append(nd.label)

            flows = solph.views.node(results, bus_label)["sequences"]

            # draw an arrow from the component to the bus
            for component in bus.inputs:
                if component.label not in labels:
                    labels.append(component.label)

                sources.append(labels.index(component.label))
                targets.append(labels.index(bus_label))

                try:
                    val = flows[
                        (
                            (component.label, bus_label),
                            (component.label, bus_label, "flow"),
                        )
                    ].sum()
                except Exception as e:
                    val = flows[((component.label, bus_label), "flow")].sum()

                if ts is not None:
                    try:
                        val = flows[
                            (
                                (component.label, bus_label),
                                (component.label, bus_label, "flow"),
                            )
                        ][ts]
                    except:
                        val = flows[((component.label, bus_label), "flow")][ts]
                # if val == 0:
                #     val = 1
                values.append(val)

            for component in bus.outputs:
                # draw an arrow from the bus to the component
                if component.label not in labels:
                    labels.append(component.label)

                sources.append(labels.index(bus_label))
                targets.append(labels.index(component.label))

                try:
                    val = flows[
                        (
                            (bus_label, component.label),
                            (bus_label, component.label, "flow"),
                        )
                    ].sum()
                except Exception as e:
                    val = flows[((bus_label, component.label), "flow")].sum()
                if ts is not None:
                    try:
                        val = flows[
                            (
                                (bus_label, component.label),
                                (bus_label, component.label, "flow"),
                            )
                        ][ts]
                    except:
                        val = flows[
                            (
                                (bus_label, component.label),
                                "flow",
                            )
                        ][ts]
                values.append(val)

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
                    source=sources,  # indices correspond to labels, eg A1, A2, A2, B1, ...
                    target=targets,
                    value=values,
                    hovertemplate="Link from node %{source.label}<br />"
                    + "to node%{target.label}<br />has value %{value}"
                    + "<br />and data <extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    return fig.to_dict()


def prepare_app(energy_system, dp_path, tables, units=None):

    # TODO to display energy system
    energy_system_graph = f"energy_system.png"
    # if ES_GRAPH is True:
    #     es = ESGraphRenderer(
    #     energy_system, legend=True, filepath=energy_system_graph, img_format="png"
    #     )
    #     es.render()
    #     energy_system_graph = encode_image_file(f"energy_system.png")

    results = energy_system.results

    bus_figures = []

    p0 = Package(dp_path)
    bus_data = pd.DataFrame.from_records(p0.get_resource("bus").read(keyed=True))
    busses = bus_data.name.tolist()

    print("busses", busses)
    date_time_index = energy_system.timeindex

    for bus in busses:
        if bus != "battery":
            fig = go.Figure(layout=dict(title=f"{bus} bus node"))
            for t, g in solph.views.node(results, node=bus)["sequences"].items():
                idx_asset = abs(t[0].index(bus) - 1)

                fig.add_trace(
                    go.Scatter(
                        x=g.index, y=g.values * pow(-1, idx_asset), name=t[0][idx_asset]
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
        df = tables[table].reset_index()
        if "Investments" in df.columns:
            df = df.loc[df.Investments > 0, ["asset", "Investments"]]
            df.rename(
                columns={"asset": "component name", "Investments": "optimized value"},
                inplace=True,
            )

        def set_value(row_number, assigned_value):
            return assigned_value.get(row_number, None)

        df["unit"] = df[df.columns[0]].apply(set_value, args=(units,))

        tables_figure.append(html.H4(table))
        tables_figure.append(
            dash_table.DataTable(
                df.to_dict("records"), [{"name": i, "id": i} for i in df.columns]
            )
        )

    # loading external resources
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    options = dict(
        # external_stylesheets=external_stylesheets
    )

    demo_app = dash.Dash(__name__, **options)

    demo_app.layout = html.Div(
        children=[
            html.H3("Scalar results"),
            html.Div(tables_figure),
            # html.Div(
            #     children=[
            #         html.P(f"{param.title()}: {settings[param]}")
            #         for param in settings.index
            #         if param != "port"
            #     ],
            #     style={"display": "flex", "justify-content": "space-evenly"},
            # ),
            # html.Div(
            #     children=dash_table.DataTable(
            #         df_costs.reset_index().to_dict("records"),
            #         [{"name": i, "id": i} for i in df_costs.reset_index().columns],
            #     )
            # ),
            # html.Div(children=[html.H3("Results in numbers"), result_div]),
            # html.Div(
            #     children=[
            #         html.H3("Non critical demand reduction overview"),
            #         dcc.Graph(
            #             id="nc_demand_supply", figure=reduced_demand_fig(results)
            #         ),
            #     ]
            # ),
            # html.Div(
            #     children=dash_table.DataTable(
            #         asset_results.reset_index().to_dict("records"),
            #         [{"name": i, "id": i} for i in asset_results.reset_index().columns],
            #     )
            # ),
            html.H3("Dynamic results"),
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
            dcc.Graph(id="sankey", figure=sankey(energy_system)),
        ]
        + [
            dcc.Graph(
                id=f"{bus}-id",
                figure=fig,
            )
            for bus, fig in zip(busses, bus_figures)
        ]
        + [dcc.Graph(id="sankey_aggregate", figure=sankey(energy_system))]
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

    @demo_app.callback(
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
        # see if case changes, otherwise do not rerun this
        date_time_index = energy_system.timeindex

        bus_figures = []
        for bus in busses:
            if bus != "battery":
                fig = go.Figure(layout=dict(title=f"{bus} bus node"))
                max_y = 0
                for t, g in solph.views.node(results, node=bus)["sequences"].items():
                    idx_asset = abs(t[0].index(bus) - 1)
                    asset_name = t[0][idx_asset]
                    if t[0][idx_asset] == "battery":
                        if idx_asset == 0:
                            asset_name += " discharge"
                        else:
                            asset_name += " charge"
                    opts = {}
                    negative_sign = pow(-1, idx_asset)
                    opts["stackgroup"] = (
                        "negative_sign" if negative_sign < 0 else "positive_sign"
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=g.index,
                            y=g.values * negative_sign,
                            name=asset_name,
                            **opts,
                        )
                    )
                    if g.max() > max_y:
                        max_y = g.max()
            else:
                capacity_battery = asset_results.capacity.battery
                if capacity_battery != 0:
                    soc_battery = (
                        solph.views.node(results, node=bus)["sequences"][
                            (("battery", "None"), "storage_content")
                        ]
                        / capacity_battery
                    )
                else:
                    soc_battery = solph.views.node(results, node=bus)["sequences"][
                        (("battery", "None"), "storage_content")
                    ]

                fig = go.Figure(layout=dict(title=f"{bus} node", yaxis_range=[0, 1]))

                fig.add_trace(
                    go.Scatter(
                        x=soc_battery.index, y=soc_battery.values, name="soc battery"
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=soc_battery.index,
                        y=np.ones(len(soc_battery.index)) * settings.storage_soc_min,
                        name="min soc battery",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=soc_battery.index,
                        y=np.ones(len(soc_battery.index)) * settings.storage_soc_max,
                        name="max soc battery",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[date_time_index[ts], date_time_index[ts]],
                    y=[0, max_y],
                    name="none",
                    line_color="black",
                )
            )
            bus_figures.append(fig)

        return [
            sankey(energy_system, date_time_index[ts]),
        ] + bus_figures

    @demo_app.callback(
        # The value of these components of the layout will be changed by this callback
        Output(component_id="ts_slice_select", component_property="value"),
        # Triggers the callback when the value of one of these components of the layout is changed
        Input(component_id="ts_slice_slider", component_property="value"),
    )
    def change_ts_value(val):
        return val

    return demo_app

    # import ipdb;ipdb.set_trace()
