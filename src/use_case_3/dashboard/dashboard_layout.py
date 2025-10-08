"""Layout functions for the dashboard"""

import dash_bootstrap_components as dbc
import yaml
from dash import html

with open("src/use_case_3/dashboard/variable_mappings.yml") as f:
    config_dict = yaml.safe_load(f)


def define_navbar_pages():
    return [
        dbc.DropdownMenuItem("More pages", header=True),
        dbc.DropdownMenuItem("Insights", href="/insights"),
        dbc.DropdownMenuItem("Ice shelf summary", href="/ice-shelf-summary"),
        dbc.DropdownMenuItem("Ice shelf analysis", href="/covariate-analysis"),
    ]


def layout_navbar(page_title):
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink(page_title, href="#")),
            dbc.DropdownMenu(
                children=define_navbar_pages(),
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand=page_title,
        brand_href="#",
        color="black",
        dark=True,
        brand_style={
            "font-size": "2rem",
            "text-align": "center",
            "font-weight": "bold",
            "margin": "auto",
        },
        sticky="sticky",
        fixed="sticky",
    )


def create_switch_button_group(dataset_id):
    # vars = list(config_dict[dataset_id.lower()]["variables"].keys())
    vars = list(config_dict[dataset_id]["variables"].keys())

    elements = [
        dbc.Switch(
            # id=f"ds_{dataset_id.lower()}",
            id=f"ds_{dataset_id}",
            label=config_dict[dataset_id]["label"],
            # id={"type": "data_switch", "index": f"ds_{dataset_id.lower()}"},
            # label=config_dict[dataset_id.lower()]["label"],
            value=False,
        ),
        html.Div(
            [
                html.P(
                    # config_dict[dataset_id.lower()]["description"],
                    config_dict[dataset_id]["description"],
                    className="analysis-data-desc",
                )
            ]
        ),
    ]

    elements_vars = []
    for var in vars:
        elements_vars.append(
            dbc.Checkbox(
                # id={"type": "data_selector", "index": dataset_id.lower() + "-" + var},
                id={"type": "data_selector", "index": dataset_id + "-" + var},
                label=config_dict[dataset_id]["variables"][var]["label"],
                # label=config_dict[dataset_id.lower()]["variables"][var]["label"],
                value=False,
            ),
        )

    if len(elements_vars) > 0:
        elements.append(
            html.Div(elements_vars, className="analysis-vars-switches", id=f"switch-group-vars-{dataset_id}")
            # html.Div(elements_vars, className="analysis-vars-switches", id=f"switch-group-vars-{dataset_id.lower()}")
            # html.Div(
            #     elements_vars,
            #     className="analysis-vars-switches",
            #     id={"type": "switch-group-vars", "index": f"switch-group-vars-{dataset_id.lower()}"},
            # )
        )

    # return html.Div(elements, id=f"switch-group-{dataset_id.lower()}")
    return html.Div(elements, id=f"switch-group-{dataset_id}")


# def create_switch_button_group_moorings(dataset_id):
#     vars = list(config_dict[dataset_id.lower()]["variables"].keys())

#     elements = [
#         dbc.Switch(
#             id=f"ds_{dataset_id.lower()}",
#             # id={"type": "data_switch", "index": f"ds_{dataset_id.lower()}"},
#             label=config_dict[dataset_id.lower()]["label"],
#             value=False,
#         ),
#         html.Div(
#             [
#                 html.P(
#                     config_dict[dataset_id.lower()]["description"],
#                     className="analysis-data-desc",
#                 )
#             ]
#         ),
#     ]

#     elements_vars = []
#     for var in vars:
#         elements_vars.append(
#             dbc.Checkbox(
#                 id={"type": "data_selector", "index": dataset_id.lower() + "-" + var},
#                 label=config_dict[dataset_id.lower()]["variables"][var]["label"],
#                 value=False,
#             ),
#         )

#     if len(elements_vars) > 0:
#         elements.append(
#             html.Div(elements_vars, className="analysis-vars-switches", id=f"switch-group-vars-{dataset_id.lower()}")
#             # html.Div(
#             #     elements_vars,
#             #     className="analysis-vars-switches",
#             #     id={"type": "switch-group-vars", "index": f"switch-group-vars-{dataset_id.lower()}"},
#             # )
#         )

#     return html.Div(elements, id=f"switch-group-{dataset_id.lower()}")
