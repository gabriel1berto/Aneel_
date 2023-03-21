from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": [ "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# PE ESSA PORRTA TA FILTRANDO A BAHIA???
df['Mes'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).month
df['Ano'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).year

counts = df.groupby(['Ano', 'Mes', 'SigUF', 'DscClasseConsumo']).count().reset_index()

fig = px.scatter(counts, x='Mes', y='NumCPFCNPJ', color='SigUF', symbol='DscClasseConsumo', animation_frame='Ano',
                 range_x=[1, 12], range_y=[-10, 200], color_discrete_sequence=px.colors.qualitative.Dark24)

fig.update_layout(
    xaxis_title='Mês',
    yaxis_title='Número de empreendimentos',
    title={
        'text': 'Número de empreendimentos por mês e UF',
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_tickfont=dict(size=14),
    yaxis_tickfont=dict(size=14),
    template='plotly_white'
)

# layout da página
app.layout = html.Div(children=[
    html.H1(children='Hello lubs'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    html.H2(children='PE ESSA PORRTA TA FILTRANDO A BAHIA???'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
    