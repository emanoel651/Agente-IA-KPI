import plotly.express as px

def line_chart(df, x, y):
    return px.line(df, x=x, y=y, markers=True)
