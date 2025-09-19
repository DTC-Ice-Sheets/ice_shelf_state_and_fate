import dash

from use_case_3.dashboard.dashboard_layout import layout_navbar

dash.register_page(__name__)

navbar = layout_navbar(page_title="Covariate Analysis")

layout = [
    navbar,
]
