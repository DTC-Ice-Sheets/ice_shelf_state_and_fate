import dash
import dash_bootstrap_components as dbc
from dash import html

from use_case_3.dashboard.dashboard_layout import layout_navbar

dash.register_page(__name__, path="/")

navbar = layout_navbar(page_title="")

layout = html.Div(
    [
        navbar,
        html.Div(
            html.Img(src="assets/DigitalTwinLogoBlueFINAL13052025.png", className="center"),
        ),
        html.Div(
            [
                html.P(
                    "Welcome to Digital Twin of Antarctic Ice Shelves. The twin is based on Earth Observation, In-situ, re-analysis, and simulation data, to provide a reconstruction of the state of Antarctic Ice Shelves, and of their interactions with the ice sheet, atmosphere, and ocean. The twin makes use of these data to forecast ice shelves evolution."
                ),
                html.P(
                    "The DTC Ice Sheets Project is part of the European Space Agency's Digital Twin Earth (DTE) initiative, developed under the Lead Digital Twin Components (DTC) Development Actions for the Ice Sheets."
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Enter the Digital Twin",
                            color="primary",
                            id="btn-homepage",
                            href="/insights",
                        ),
                    ],
                    className="d-grid gap-2 col-6 mx-auto",
                ),
                html.P(""),
                # html.P(
                #     "TODO: Add consortium member logos and a better description and remove unnecessary white space around logo."
                # ),
            ],
            className="website_text",
        ),
    ]
)
