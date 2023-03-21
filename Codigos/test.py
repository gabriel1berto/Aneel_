import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# Carrega os gráficos do Desafio 1.inpn
app = dash.Dash(__name__)

fig_style = {'padding': '0 20px'}

# Carrega os gráficos do Desafio 1.inpy
with open('emp_mes_ufF.html', 'r') as f:
    fig1 = html.Iframe(srcDoc=open('emp_mes_ufF.html', 'r', encoding='utf-8').read(), width='100%', height='600')
with open('emp_por_ano.html', 'r') as f:
    fig2 = html.Iframe(srcDoc=open('emp_por_ano.html', 'r', encoding='utf-8').read(), width='100%', height='600')


app.layout = html.Div([
    html.Img(src="/sites/default/files/Logo.svg", title="Energisa", alt="Logo Energisa",
             style={
                 'position': 'absolute', # posição absoluta
                 'left': '10px', # distância da borda esquerda
                 'top': '10px', # distância da borda superior
                 'height': '50px' # altura da imagem
             }),

    # Título da página
    html.H1(children='Desafio Técnico Engergisa', 
            style={
                'color': '#0074D9', # cor do texto
                'font-size': '36px', # tamanho da fonte
                'font-family': 'Arial, sans-serif', # família da fonte
                'font-weight': 'bold', # negrito
                'margin-top': '50px', # margem superior
                'margin-bottom': '30px', # margem inferior
                'text-align': 'center' # alinhamento do texto
            }),

    # Descrição da página
    html.Div(children='''
        OBS: Estou muito animado para compartilhar estes gráficos com você!
    '''),

    # Local onde os gráficos atualizados serão exibidos
    html.Div(children=[
        html.H2(children='Gráfico atualizado'),
        dcc.Graph(id='graph1')
    ]),

    # Define as funções para atualização dos gráficos
    html.Div(children=[
        html.H2(children='Gráfico atualizado'),
        dcc.Graph(id='graph1')
    ], style={'backgroundColor': '#FFF5EE'})
])

@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [dash.dependencies.Input('dropdown1', 'value'),
     dash.dependencies.Input('slider1', 'value'),
     dash.dependencies.Input('button1', 'n_clicks')])
def update_graph1(dropdown_value, slider_value, n_clicks):
    # Código para atualizar o gráfico 1 com base nos valores do dropdown, slider e botão
    # Se for necessário, adicione as variáveis dropdown_value, slider_value e n_clicks na sua lógica de atualização do gráfico
    
    # Exemplo de atualização do gráfico
    data = px.data.gapminder()
    filtered_data = data[data['year'] == slider_value]
    fig = px.scatter(filtered_data, x='gdpPercap', y='lifeExp', color='continent', title=f'Ano {slider_value}')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
