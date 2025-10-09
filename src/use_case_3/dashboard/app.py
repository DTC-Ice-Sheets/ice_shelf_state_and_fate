"""Dashboard entrypoint."""

import logging

import dash
import dash_bootstrap_components as dbc
import diskcache
import dotenv
from dash import Dash, DiskcacheManager, html

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    use_pages=True,
    background_callback_manager=background_callback_manager,
)

app.layout = html.Div(
    [
        dash.page_container,
    ]
)


if __name__ == "__main__":
    app.run(debug=True)
