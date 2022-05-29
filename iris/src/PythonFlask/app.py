from importlib.resources import path
from dash import Dash, Input, Output, dcc, html
from flask import Flask, render_template
import plotly.express as px
import dash_bootstrap_components as dbc
import markdown

import pages.timeline_dash.index as dtimeline
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
    "margin-left": "16rem"
}

sidebar = html.Div(
    [
        html.H2("Analytics", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Timeline", href="/timeline", active="exact"),
                dbc.NavLink("IRIS BI", href="/irisbi", active="exact"),
                dbc.NavLink("Logi report", href="/logi_report", active="exact"),
                dbc.NavLink("PowerBI report", href="/powerbi_report", active="exact"),
                dbc.NavLink("Tableau report", href="/tableau_report", active="exact")
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
        with open('/irisdev/app/README.md', 'r') as f:
            return html.Article(dcc.Markdown(f.read()), className="markdown-body")

    elif pathname == "/timeline":
        return html.Div(dtimeline.GetFigure())

    elif pathname == "/logi_report":
        return html.Iframe(src="assets/All_countries.pdf", 
            style={"height" : "100vh", "width": "100%", "display" : "flex"}
        )

    elif pathname == "/powerbi_report":
        return html.Iframe(
            src="https://app.powerbi.com/view?r=eyJrIjoiYzYwYTZmZjYtM2E3Mi00NWRjLTk4MmEtN2FhODkxNTU0N2ZjIiwidCI6ImMwNDU1OGJhLWJiMzgtNDQzMC1iMDhkLThlMTYxMmQzY2NkOCIsImMiOjl9",
            style={"height" : "100vh", "width": "100%", "display" : "flex"}
        )

    elif pathname == "/tableau_report":
        return html.Iframe(src="assets/Water Conditions in Europe.pdf", 
            style={"height" : "100vh", "width": "100%", "display" : "flex"}
        )

    elif pathname == "/irisbi":
        return html.Iframe(
            src="http://localhost:32792/dsw/index.html#/IRISAPP/dc/teccod/dashboards/Overview.dashboard",
            style={"height" : "100vh", "width": "100%", "display" : "flex"}
        )

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=False, port=8080, host="0.0.0.0")