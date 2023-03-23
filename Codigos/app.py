# importa bibliotecas
import requests
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import dcc
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff


############################################################ API ##############################################################################

# define url incrementada para requisição de api (ate 1.000.000 registros)
url = "https://dadosabertos.aneel.gov.br/api/action/datastore_search?resource_id=b1bd71e7-d0ad-4214-9053-cbd58e9564a7&limit=1000000"

# tempo máximo de espera para a resposta da requisição
timeout_seconds = 100   

# Obter os dados da API
def get_data(url, timeout_seconds):
    try:
        # Requisição GET com o tempo limite definido
        response = requests.get(url, timeout=timeout_seconds)

        #  Verifica erros HTTP
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Erro ao obter os dados:", e)
        return None

    # Código de status da resposta HTTP=200, indicando que a requisição foi bem sucedida
    if response.status_code == 200:
        # Em caso de sucesso, obtenha os dados em formato JSON e imprima "Requisição bem sucedida"
        data = response.json()
        print("Requisição bem sucedida")

        # Verificar resposta da API
        if data["success"]:
            # Gera lista de registros e a lista de colunas
            records = data["result"]["records"]
            columns = data["result"]["fields"]

            # Adiciona as colunas "Bandeira" e "Nome_agente" na lista de colunas
            columns.append({"id": "Bandeira", "type": "text"})
            columns.append({"id": "Nome_agente", "type": "text"})

            # Cria o DataFrame
            df = pd.DataFrame.from_records(records, columns=[c["id"] for c in columns])

            # Retorne o DataFrame criado
            return df
        else:
            # Em caso de erro, imprir uma mensagem de aviso e retorne None
            print("A API não retornou dados.")
            return None
    else:
        # Se o código de status da resposta HTTP for diferente de 200, trate o erro de acordo com o código
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

# Chame a função get_data para obter um DataFrame com os dados da API
df = get_data(url, timeout_seconds)


############################################################ GRÁFICO 1 ##############################################################################


# Análise de dados para elaboração de um gráfico mostrando a evolução temporal do Total Mensal de Empreendimentos por Estado e por Classe de Consumo;

    # OBS1: estados com menos de 10 empreendimentos foram excluídos do gráfico;
    # OBS2: os dados de 2023 representam um corte até hoje

df['Mes'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).month
df['Ano'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).year

# cria uma cópia do DataFrame sem valores nulos de NumCPFCNPJ
df_clean = df.dropna(subset=['NumCPFCNPJ']).copy()

# filtra apenas as SigUFs desejadas
df_clean = df_clean[df_clean['SigUF'].isin(['BA', 'AL', 'AC', 'AM', 'AP'])]

# agrupa os dados por ano, mês, SigUF e DscClasseConsumo
counts = df_clean.groupby(['Ano', 'SigUF', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()

import numpy as np

# cria o gráfico de áreas empilhadas animado
fig_1 = px.bar(counts, x="Ano", y="NumCPFCNPJ", color="DscClasseConsumo", animation_frame="SigUF", 
               title="Novos usuários da Energia distribuida")

# calcula o valor máximo dos dados do eixo y
y_max = counts['NumCPFCNPJ'].max()

# Define a faixa de valores do eixo y de acordo com os dados do gráfico
fig_1.update_yaxes(range=[0, 7000])


############################################################ GRÁFICO 2 ##############################################################################


df['Mes'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).month
df['Ano'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).year

counts = df_clean.groupby(['Ano', 'SigUF', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()
tabela = counts.groupby(['Ano', 'SigUF'])['NumCPFCNPJ'].sum().reset_index()
matriz = tabela.pivot(index='SigUF', columns='Ano', values='NumCPFCNPJ')

matriz['Total'] = matriz.sum(axis=1)
matriz = matriz.fillna(0)

matriz_tx = matriz.iloc[:, :-1].pct_change(axis=1) * 100
matriz_tx= matriz.applymap(lambda x: '{:.0f}%'.format(x) if not np.isinf(x) else '∞')
matriz_tx['med'] =  0
matriz_tx['2023'] = 0

# Criando uma lista de strings com valores da matriz e matriz_tx
text = [[f"{matriz.iloc[i,j]:.0f}%<br>{matriz_tx.iloc[i,j]}" for j in range(len(matriz.columns))] for i in range(len(matriz.index))]

# Criação do heatmap principal
fig = ff.create_annotated_heatmap(z=matriz.values,
                                  x=list(matriz.columns),
                                  y=list(matriz.index),
                                  annotation_text=text,
                                  font_colors=['gray', 'white'],
                                  colorscale='YlGnBu')

#Configurações de layout
fig.update_layout(
    width=1000,
    height=500,
    xaxis_title='Ano',
    xaxis_title_font_size=16,
    xaxis_tickfont_size=14,
    yaxis_title='SigUF',
    yaxis_title_font_size=16,
    yaxis_tickfont_size=14,
    title='Novos usuários da Energia distribuida',
    title_font_size=30,
    title_x=.5,
    margin=dict(t=130)
)


############################################################ GRÁFICO 3 ##############################################################################

df_prophet = pd.read_excel("prophet_aneel.xlsx")
df_prophet.head()

df_prophet_ = df_prophet[['ds','yhat','y_real', 'yhat_upper', 'yhat_lower']]

# fig_2 = px.line(df_prophet[['ds','yhat','y_real']], x="ds", y=["yhat",'y_real'],title="Novos usuários da Energia distribuida")
# fig_2.show()

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

layout = go.Layout(title='Projeção de quantidade de CPF', xaxis=dict(title='Eixo X'), yaxis=dict(title='Eixo Y'))

fig_prophet = go.Figure(data=data, layout=layout)

#pyo.plot(fig_prophet, filename='grafico.html')



############################################################ DASH ##############################################################################


app = dash.Dash(__name__)


app.layout = html.Div(children=[
    html.H1(children='Meu Aneel'),
    
    html.Div(children='''
        Aqui está o meu dashboard!
    '''),


    html.Div([
        dcc.Graph(
            id='grafico1',
            figure=fig_1,
            style={'width': '50%', 'display': 'inline-block'}
        ),

        dcc.Graph(
            id='grafico2',
            figure=fig_1,
            style={'width': '50%', 'display': 'inline-block'}
        ),
    
    ]),
    html.Div([
        dcc.Graph(
            id='grafico1',
            figure=fig_1,
            style={'width': '50%', 'display': 'inline-block'}
        ),

        dcc.Graph(
            id='grafico2',
            figure=fig_1,
            style={'width': '50%', 'display': 'inline-block'}
        ),
    
    ]),

    html.Div([
        dcc.Graph(
            id='grafico3',
            figure=fig_prophet,
            style={'width': '70%', 'display': 'inline-block'}
        ),

    ]),

])

    

if __name__ == '__main__':
    app.run_server(debug=True)