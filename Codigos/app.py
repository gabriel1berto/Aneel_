# Bibliotecas - API
import requests

# Bibliotecas - Gráficos
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Bibliotecas - Dash
import dash
from dash import dcc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output




############################################################ API ##############################################################################



# define url incrementada para requisição de api (ate 1.000.000 registros)
url = "https://dadosabertos.aneel.gov.br/api/action/datastore_search?resource_id=b1bd71e7-d0ad-4214-9053-cbd58e9564a7&limit=1000000"

# tempo máximo de espera para a resposta da requisição
timeout_seconds = 100   

# obter os dados da API
def get_data(url, timeout_seconds):
    try:
        # requisição GET com o tempo limite definido
        response = requests.get(url, timeout=timeout_seconds)

        # verifica erros HTTP
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Erro ao obter os dados:", e)
        return None

    ## HTTP=200 indica que a requisição foi bem sucedida
    if response.status_code == 200:
        # Em caso de sucesso, obtenha os dados em formato JSON e imprime"Requisição bem sucedida"
        data = response.json()
        print("Requisição bem sucedida")

        # verificar resposta da API
        if data["success"]:
            # gera lista de registros e a lista de colunas
            records = data["result"]["records"]
            columns = data["result"]["fields"]

            # adiciona as colunas "Bandeira" e "Nome_agente" na lista de colunas
            columns.append({"id": "Bandeira", "type": "text"})
            columns.append({"id": "Nome_agente", "type": "text"})

            # cria o DataFrame
            df = pd.DataFrame.from_records(records, columns=[c["id"] for c in columns])

            # retorne o DataFrame criado
            return df
        else:
            # rm caso de erro, imprir uma mensagem de aviso e retorne None
            print("A API não retornou dados.")
            return None
    else:
        # se o código de status da resposta HTTP for diferente de 200, trate o erro de acordo com o código
        if response.status_code == 400:
            print("Requisição mal formada.")
        elif response.status_code == 401:
            print("Não autorizado.")
        elif response.status_code == 403:
            print("Acesso proibido.")
        elif response.status_code == 404:
            print("Recurso não encontrado.") 
        elif response.status_code == 500:
            print("Erro interno do servidor.")
        else:
            print("Erro desconhecido ao obter os dados.")

        return None

# chama get_data para obter df
df = get_data(url, timeout_seconds)

# visualiza dados
# df = pd.read_excel("dados.xlsx")



############################################################ GRÁFICO 1 ##############################################################################



# O gráfico 1 mostra a evolução temporal do Total Mensal de Empreendimentos por Estado e por Classe de Consumo;

    # OBS1: estados com menos de 10 empreendimentos foram excluídos do gráfico;
    # OBS2: os dados de 2023 representam um corte até hoje


df['Mes'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).month
df['Ano'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).year
df['Ano_Mes'] = pd.to_datetime(df['DthAtualizaCadastralEmpreend']).dt.strftime('%Y-%m')

# exclui nulos
df_clean = df.dropna(subset=['NumCPFCNPJ']).copy()

# filtra SigUFs 
df_clean = df_clean[df_clean['SigUF'].isin(['BA', 'AL', 'AC', 'AM', 'AP'])]

# agrupa os dados
counts = df_clean.groupby(['Ano_Mes', 'SigUF', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()

# Cria uma lista com as opções de SigUF para dropdown
sigufs = df_clean['SigUF'].unique()

# cria o gráfico de áreas empilhadas

fig_3 = px.bar(counts, x="Ano_Mes", y="NumCPFCNPJ", color="DscClasseConsumo", title="Novos usuários da rede distribuida")

# estilo do grafico
fig_3.update_layout(
    width=1200,
    height=600,
    xaxis_title='Ano_Mes',
    xaxis_title_font_size=16,
    xaxis_tickfont_size=14,
    yaxis_title='NumCPFCNPJ',
    yaxis_title_font_size=16,
    yaxis_tickfont_size=14,
    title='Novos usuários da Energia distribuida',
    title_font_size=20,
    title_x=.5,
    margin=dict(t=130)
)


fig_3.update_yaxes(range=[0, 6000])



############################################################ GRÁFICO 2 ##############################################################################


# O gráfico 2 mostra a evolução temporal do Total Mensal de Empreendimentos por Estado e por Classe de Consumo;


df['Mes'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).month
df['Ano'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).year

counts = df_clean.groupby(['Ano', 'SigUF', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()
tabela = counts.groupby(['Ano', 'SigUF'])['NumCPFCNPJ'].sum().reset_index()

# manioula a tabela para gerar a matriz
matriz = tabela.pivot(index='SigUF', columns='Ano', values='NumCPFCNPJ')
matriz = matriz.fillna(0)
matriz.loc['BR'] = matriz.sum()

# manipula a matriz para gerar a matriz_tx
matriz_tx = matriz.pct_change(axis=1) * 100
matriz_tx = matriz_tx.applymap(lambda x: '{:.0f}%'.format(x) if not np.isnan(x) and x != 0 else ' ')
matriz_tx['med'] = ""
matriz_tx['Total'] = ""
matriz_tx['2023'] = ""


# lista de strings com valores da matriz e matriz_tx
text = [[f"{matriz.iloc[i,j]:}<br>{matriz_tx.iloc[i,j]}" for j in range(len(matriz.columns))] for i in range(len(matriz.index))]

# Heatmap principal
fig_2 = ff.create_annotated_heatmap(z=matriz.values,
                                  x=list(matriz.columns),
                                  y=list(matriz.index),
                                  annotation_text=text,
                                  font_colors=['gray', 'white'],
                                  colorscale='YlGnBu')

# estilo do heatmap
fig_2.update_layout(
    width=1200,
    height=600,
    xaxis_title='Ano',
    xaxis_title_font_size=16,
    xaxis_tickfont_size=14,
    yaxis_title='SigUF',
    yaxis_title_font_size=16,
    yaxis_tickfont_size=14,
    title='Novos usuários da Energia distribuida',
    title_font_size=20,
    title_x=0.5,
    margin=dict(t=100)
)




############################################################ GRÁFICO 3 ##############################################################################



# o gráfico 3 é um gráfico de linha com a previsão de novos usuários da energia distribuida

# OBS: por conflito de bibliotecas o gráfico foi gerado em um arquivo separado no colab e importado para o streamlit.


df_prophet = pd.read_excel("prophet_aneel.xlsx")
df_prophet.head()

df_prophet_ = df_prophet[['ds','yhat','y_real', 'yhat_upper', 'yhat_lower']]

# fig_prophet = px.line(df_prophet[['ds','yhat','y_real']], x="ds", y=["yhat",'y_real'],title="Novos usuários da Energia distribuida")
# fig_prophet.show()

import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

x = df_prophet_['ds']
y1 = df_prophet_['yhat']
y2 = df_prophet_['y_real'] 
y3 = df_prophet_['yhat_upper']
y4 = df_prophet_['yhat_lower']

trace1 = go.Scatter(x=x, y=y1, mode='lines', name='Previsto', line=dict(color='blue'))
trace2 = go.Scatter(x=x, y=y2, mode='markers', name='Real',marker=dict(color='black'))
trace3 = go.Scatter(x=x, y=y3, fill=None, mode='lines', line_color='gray', name='Intervalo Superior')
trace4 = go.Scatter(x=x, y=y4, fill='tonexty', mode='lines', line_color='gray', name='Intervalo Inferior')

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(title=dict(text='Projeção de usuários da rede distribuida no Brasil', x=0.5), xaxis=dict(title='Eixo X'), yaxis=dict(title='Eixo Y'))

fig_prophet = go.Figure(data=data, layout=layout)

#pyo.plot(fig_prophet, filename='grafico.html')




############################################################ DASH ##############################################################################


# inicia o app
app = dash.Dash(__name__)

# cria o dropdown com as opções de estados
dropdown = dcc.Dropdown(
    id='dropdown',
    options = sigufs,
    value='AC',
    style={'padding-left': '30px', 'width': '50%'}
)

# atualiza o gráfico com base no valor selecionado no dropdown
@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('dropdown', 'value')]
)

def update_figure(selected_option):
    counts_ = counts[counts['SigUF'] == selected_option]
    fig_3 = px.bar(counts_, x="Ano", y="NumCPFCNPJ", color="DscClasseConsumo", title="Novos usuários da Energia distribuida")
    return fig_3


#__________________________________LAYOUT_____________________________________________________
app.layout = html.Div(children=[
       

    html.Img(
        src='https://i2.energisa.info/SiteAssets/topo-marca.png',
        style={
            'height': '70px',
            'width': 'auto',
            'float': 'right'
        }
    ),

    # Título
    html.H1(
        children='Análise da geração distribuida no Brasil', 
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '50px', 'font-weight': '700', 'line-height': '1.2', 'color': '#D75413', 'margin-top': '40px'}
    ),

    # Descrição
    html.Div(
        children=''' Análises desenvolvidas por Gabriel Humberto como desafio técnico da Energisa.''',
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '15px', 'font-weight': '700', 'line-height': '1.2', 'color': '#181818', 'margin-top': '20px'}
    ),

    html.Hr(style={'margin': '10px 0', 'border-style': 'dashed', 'border-width': '1px', 'opacity': '0.5'}),


#__________________GRÁFICOS_______________#

    html.H2('Análise de novos usuários da Energia distribuida por estado',
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '25px', 'font-weight': '700', 'line-height': '1.2', 'color': '#0774B4', 'margin-top': '20px'}
    ),
   
    html.Div([
        dropdown,
        html.Div([
            dcc.Graph(id='graph',
                    figure=fig_3,
                    style={'width': '80%', 'display': 'inline-block'}),
            html.Div(
                id='caixa-de-texto1',
                children=[
                    html.P('Objetivo e insights',
                        style={'font-size': '20px','padding-left': '10px','font-size': '23px', 'text-align': 'center','padding-right': '20px'}),
                    html.Div('Utilizando um gráfico de barras, foi possível visualizar a evolução do número de usuários da geração distribuída ao longo do tempo. A análise demonstrou um crescimento quase contínuo, com alguns outliers positivos, como o estado de Alagoas, e um outlier negativo, como a Bahia.',
                            style={'font-size': '15px','padding-left': '10px','height': '300px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'center','padding-right': '20px'})
                ],
                style={'margin-top': '-50px', 'width': '20%', 'display': 'inline-block', 'font-family': 'Roboto, Arial, sans-serif', 'font-size': '30px', 'font-weight': '700', 'line-height': '1.2', 'color': '#181818', 'margin-top': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    ]),


    html.Hr(style={'margin': '10px 0', 'border-style': 'dashed', 'border-width': '1px', 'opacity': '0.5'}),


    html.H2('Análise macro de novos usuários da Energia distribuida',
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '25px', 'font-weight': '700', 'line-height': '1.2', 'color': '#0774B4', 'margin-top': '20px'}
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='grafico2',
                    figure=fig_2,
                    style={'width': '80%', 'display': 'inline-block'}),
            html.Div(
                id='caixa-de-texto2',
                children=[
                    html.P('Objetivo e insights',
                        style={'font-size': '20px','padding-left': '10px','font-size': '23px', 'text-align': 'center','padding-right': '20px'}),
                    html.Div('O mapa de calor situa o cenário da evolução da geração distribuída no país, com ele observamos a distoante evolução de Alagoas na geraçãao distribuída no cenário nascional.',
                            style={'font-size': '15px','padding-left': '10px','height': '300px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'center','padding-right': '20px'})
                ],
                style={'margin-top': '-50px', 'width': '20%', 'display': 'inline-block', 'font-family': 'Roboto, Arial, sans-serif', 'font-size': '30px', 'font-weight': '700', 'line-height': '1.2', 'color': '#181818', 'margin-top': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    ]),





    html.Hr(style={'margin': '10px 0', 'border-style': 'dashed', 'border-width': '1px', 'opacity': '0.5'}),

    html.H2('Projeção de novos usuários da Energia distribuida',
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '25px', 'font-weight': '700', 'line-height': '1.2', 'color': '#0774B4', 'margin-top': '20px'}
    ),

    html.Div([
        html.Div([
            dcc.Graph(id='grafico3',
                    figure=fig_prophet,
                    style={'width': '70%', 'display': 'inline-block'}),
            html.Div(
                id='caixa-de-texto3',
                children=[
                    html.P('Objetivo e insights',
                        style={'font-size': '20px','padding-left': '10px','font-size': '23px', 'text-align': 'center','padding-right': '20px'}),
                    html.Div('Visando prever a evolução da entrada de usuários no sistema de geração distribuída no país, foi desenvolvido um modelo para prever os próximos 12 meses por séries temporais com a biblioteca Prophet do Python. O modelo foi treinado com dados históricos de entrada de usuários e considera fatores como sazonalidade e tendência para realizar as previsões. ',
                            style={'font-size': '18px','padding-left': '10px','height': '300px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'center','padding-right': '20px'})
                ],
                style={'margin-top': '-50px', 'width': '30%', 'display': 'inline-block', 'font-family': 'Roboto, Arial, sans-serif', 'font-size': '30px', 'font-weight': '700', 'line-height': '1.2', 'color': '#181818', 'margin-top': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    ]),

])


# http://127.0.0.1:8050/

if __name__ == '__main__':
    app.run_server(debug=True)