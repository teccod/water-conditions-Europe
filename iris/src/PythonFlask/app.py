from dash import Dash, Input, Output, dcc, html
from flask import Flask
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import markdown

import pages.timeline_dash as dtimeline
import pages.ml.ml_run as ml

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                # dbc.NavLink("Readme", href="/readme", active="exact"),
                dbc.NavLink("Timeline", href="/timeline", active="exact"),
                dbc.NavLink("Form", href="/form", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P("This is the content of the home page!")

    elif pathname == "/readme":
        with open('/irisdev/app/README.md', 'r') as f:
            return html.Article(dcc.Markdown(f.read()), className="markdown-body")

    elif pathname == "/timeline":
        return html.Div(dtimeline.GetFigure())

    elif pathname == "/form":
        return html.P(ml.test())

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8080, host="0.0.0.0")