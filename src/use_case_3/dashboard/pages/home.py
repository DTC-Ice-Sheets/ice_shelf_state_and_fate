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
                    "Welcome to the Digital Twin Ice Sheets. This dashboard allows to you to investigate the current and "
                    "future state of the Antarctic ice shelves using earth observation data. "
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
                html.P("TODO: Add consortium member logos and a better description."),
            ],
            className="website_text",
        ),
    ]
)
