import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# Carrega os gráficos do Desafio 1.inpy
with open('emp_mes_ufF.html', 'r', encoding='utf-8') as f:
    fig1 = html.Iframe(srcDoc=f.read(), width='100%', height='600')

with open('emp_por_ano.html', 'r', encoding='utf-8') as f:
    fig2 = html.Iframe(srcDoc=f.read(), width='100%', height='600')

# Define o layout da página
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    
    # Título da página
    html.H1(children='Teste Energisa'),
    
    # Descrição da página
    html.Div(children='''
        OBS: Estou muito animado para compartilhar estes gráficos com você!
    '''),
    
    # Gráficos importados
    html.Div(children=[
        html.H2(children='Gráfico 1'),
        fig1,
        
        html.H2(children='Gráfico 2'),
        fig2
    ]),
    
    # Elementos interativos
    html.Div(children=[
        
        html.H2(children='Elementos interativos'),
        
        # Dropdown: é um menu suspenso que permite ao usuário selecionar uma das opções disponíveis.
        dcc.Dropdown(     
            id='dropdown1',
            options=[
                {'label': 'Opção 1', 'value': 'opcao1'},
                {'label': 'Opção 2', 'value': 'opcao2'},
                {'label': 'Opção 3', 'value': 'opcao3'}
            ],
            value='opcao1'
        ),
        
        # Slider: é uma barra deslizante que permite ao usuário selecionar um valor dentro de um intervalo definido. 
        dcc.Slider(
            id='slider1',
            min=0,
            max=10,
            step=1,
            value=5
        ),
        

        # Button: é um botão que o usuário pode clicar para executar uma ação.
        html.Button('Botão 1', id='button1')
    ]),
    
    # Local onde os gráficos atualizados serão exibidos
    html.Div(children=[
        html.H2(children='Gráfico atualizado'),
        dcc.Graph(id='graph1')
    ])
])

# Define as funções para atualização dos gráficos
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
    