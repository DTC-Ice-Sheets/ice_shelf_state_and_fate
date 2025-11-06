"""Dashboard entrypoint."""

import logging
import os

import dash
import dash_auth
import dash_bootstrap_components as dbc
import diskcache
import dotenv
from dash import Dash, DiskcacheManager, html

from use_case_3.dashboard.dashboard_layout import url_base_pathname

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=True,
    background_callback_manager=background_callback_manager,
    url_base_pathname=url_base_pathname,
)

auth = dash_auth.BasicAuth(
    app,
    {"": os.environ["DASHBOARD_PASSWORD"]},
    secret_key=os.environ["DASHBOARD_PASSWORD"],
)

app.layout = html.Div(
    [
        dash.page_container,
    ]
)

server = app.server

if __name__ == "__main__":
    app.run(debug=False)
