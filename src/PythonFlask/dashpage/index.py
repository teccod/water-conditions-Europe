import iris
import pandas as pd
import plotly.express as px

query = """
    select 
        dc.alpha_3,
        dc.name as country,
        dw.phenomenonTimeReferenceYear, 
        AVG(resultMeanValue) as resultmeanvalue
    FROM dc_data_teccod.waterPollution dw
    LEFT JOIN dc_data_teccod.Countries dc on dw.country = dc.name
    WHERE dw.procedureAnalysedMedia = 'water'
"""

df = iris.sql.exec(query).dataframe()
df = df.sort_values(by=['phenomenontimereferenceyear'], ascending=True)

def GetFigure():
    return px.choropleth(df, locations="alpha_3",
        scope="europe",
        color="resultmeanvalue",
        hover_name="alpha_3",
        animation_frame="phenomenontimereferenceyear",
        color_continuous_scale=px.colors.sequential.Plasma)