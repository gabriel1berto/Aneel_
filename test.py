import dash_html_components as html
from dash import Dash, dcc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
#import plotly.io as pio
from dash_html_components import Iframe


# Cores: (rgb(68,199,244)); rgb(37,143,185); rgb(245,145,6)

# Carrega os gráficos do Desafio 1.inpn

app = Dash(__name__)

fig_style = {'padding': '0 20px'}

# Defina o layout
app.layout = html.Div(
    style={
        'display': 'flex',
        'flex-direction': 'column',
        'align-items': 'center',
        'height': '100vh',
        'background-color': 'rgb(68, 199, 244)',
    },
    children=[
        html.Img(
            src='https://i2.energisa.info/SiteAssets/topo-marca.png',
            alt='Logo Energisa',
            title='Energisa',
            style={'height': '50px'},
        ),
        html.H1(
            children='Desafiooooo',
            style={
                'font-size': '36px',
                'font-family': 'Arial, sans-serif',
                'font-weight': 'bold',
                'color': '#0074D9',
                'margin-top': '30px',
                'text-align': 'center',
            },
        ),
        html.Div(
            children='''OBS: Estou muito animado para compartilhar estes gráficos com você!''',
            style={'margin-top': '20px'},
        ),

        html.Div(children=[
            html.H2(children='Gráfico atualizado'),
            html.Iframe(srcDoc=open('grafico.html', 'r').read(), width='100%', height='500')
        ]),

        # Define as funções para atualização dos gráficos
        html.Div(children=[
            html.H2(children='Gráfico atualizado'),
            dcc.Graph(id='graph2'),
            dcc.Graph(id='graph3')
        ], style={'backgroundColor': '#FFF5EE'})
    ],
)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)