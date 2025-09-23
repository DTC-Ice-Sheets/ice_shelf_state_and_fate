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
                    "Welcome to Digital Twin Ice Sheets. This dashboard allows to you to investigate the current and "
                    "future state of the Antarctic ice shelves using earth observation data. It provides an up-to-date reconstruction of Antarctic Ice Shelves, their interactions with the ice sheet, atmosphere, and ocean, and enables the forecast of their evolution."
                ),
                html.P(
                    "The DTC Ice Sheets Project is part of the European Space Agency’s Digital Twin Earth (DTE) initiative, developed under the Lead Digital Twin Components (DTC) Development Actions for the Ice Sheets."
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Start analysis",
                            color="primary",
                            id="btn-homepage",
                            href="/ice-shelf-summary",
                        ),
                    ],
                    className="d-grid gap-2 col-6 mx-auto",
                ),
                html.P(""),
                html.P(
                    "TODO: Add consortium member logos and a better description and remove unnecessary white space around logo."
                ),
            ],
            className="website_text",
        ),
    ]
)
