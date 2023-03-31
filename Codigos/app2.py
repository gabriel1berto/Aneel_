

############################################################ BIBLIOTECAS ##############################################################################


# Bibliotecas - API
import requests

# Bibliotecas - Gráficos
import numpy as np
import pandas as pd
import missingno as msno
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# Bibliotecas - Dash
import dash
from dash import dcc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Outras
import warnings
warnings.filterwarnings('ignore')


############################################################ API ##############################################################################


# Cria a url de requisição da API (ate 1.000.000 registros)
resource_id="b1bd71e7-d0ad-4214-9053-cbd58e9564a7"
limit_param = 1000000
timeout_seconds = 100

def criar_url(resource_id, limit):
    url_base = "https://dadosabertos.aneel.gov.br/api/action/datastore_search?"
    resource_param = "resource_id=" + resource_id
    limit_param = "limit=" + str(limit)
    url = url_base + resource_param + "&" + limit_param
    return url

url = criar_url(resource_id, limit_param)

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

            # Opção de adicionar colunas "Bandeira" e "Nome_agente" na lista de colunas
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



############################################################ EDA ##############################################################################

import matplotlib.pyplot as plt
import missingno as msno

# Atualizando formato de colunas
def atualiza_formato_colunas(df):
    df = df.astype({
        "NumCNPJDistribuidora": np.int64,
        "CodClasseConsumo": np.int64,
        "CodSubGrupoTarifario": np.int64,
        "codUFibge": np.float64,
        "codRegiao": np.float64,
        "CodMunicipioIbge": np.float64,
        "QtdUCRecebeCredito": np.int64,
    })
    return df
df = atualiza_formato_colunas(df)

# análise de vazios
def analyze_dataframe(df):
    # Análise de colunas duplicadas
    duplicated_cols = df.columns[df.columns.duplicated(keep=False)]
    df_duplicated_col = df[duplicated_cols].sum()
    print('Colunas duplicadas: ', df_duplicated_col.tolist())

    # Análise de linhas duplicadas
    duplicated_rows = df.duplicated(keep=False)
    df_duplicated_line = duplicated_rows.sum()
    print('Linhas duplicadas: ', df_duplicated_line.tolist())

    # Análise de valores nulos
    na_tot = df.isna().sum().sort_values(ascending=False)
    na_perc = (df.isna().sum() / df.shape[0] * 100).round(2).sort_values(ascending=False)
    na = pd.concat([na_tot, na_perc], axis=1, keys=['+', '%'])
    print(na.head(10))
#analyze_dataframe(df)

def check_duplicates(df):
    # Verifica colunas duplicadas
    duplicated_cols = df.columns[df.columns.duplicated(keep=False)]
    num_duplicated_cols = len(duplicated_cols)
    
    # Verifica linhas duplicadas
    duplicated_rows = df.duplicated(keep=False)
    num_duplicated_rows = duplicated_rows.sum()
    
    # Retorna o resultado
    return num_duplicated_cols, num_duplicated_rows
#check_duplicates(df)


def visualizar_nulos(df):
    sorted_df = df.sort_values(by='NumCoordEEmpreendimento')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Cria a matriz de visualização de nulos com o eixo x em rotação vertical e tamanho de fonte menor
    matriz_nulos = msno.matrix(sorted_df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.show()
#visualizar_nulos(df)


# Adicionando as colunas Ano, Mes e Ano_Mes
def adiciona_colunas(df):
    df['DatetimeIndex'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend'])
    df['Mes'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).month
    df['Ano'] = pd.DatetimeIndex(df['DthAtualizaCadastralEmpreend']).year
    df['Ano_Mes'] = pd.to_datetime(df['DthAtualizaCadastralEmpreend']).dt.strftime('%Y-%m')
    return df
adiciona_colunas(df)


###### CRIA DFS #######

# Para evitar conflitos por manipulação nas análises, criei dfs específicos para cada análise oriundos de simplificações de df
# df pase para gráficos
df_clean = df.dropna(subset=['NumCPFCNPJ']).copy()

# estudos da serie temporal
ts = df_clean.groupby(['DthAtualizaCadastralEmpreend', 'SigUF','SigTipoConsumidor', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()
ts = ts.sort_values('DthAtualizaCadastralEmpreend')
ts = ts.set_index('DthAtualizaCadastralEmpreend')
ts_clean = ts.drop(['SigUF', 'SigTipoConsumidor','DscClasseConsumo'], axis=1)
ts_clean = ts_clean.groupby('DthAtualizaCadastralEmpreend').sum()

# a cada modelo de previsão será criado um df (ts_nome-do-modelo)



############################################################ ESTUDO DOS DADOS ##############################################################################


#______________cria parametros
import statsmodels.api
from statsmodels.tsa.seasonal import seasonal_decompose
resultado = seasonal_decompose(ts_clean['NumCPFCNPJ'], period = 12)
tendencia = resultado.trend
sazonalidade = resultado.seasonal
residuo = resultado.resid

#______________tendencia
fig_tendencia = go.Figure()
# Adicionar linha da tendência
fig_tendencia.add_trace(go.Scatter(x=ts_clean.index, y=tendencia, mode='lines', name='Tendência'))

# Configurar o layout
fig_tendencia.update_layout(
    xaxis_title='DthAtualizaCadastralEmpreend',
    yaxis_title='NumCPFCNPJ',
    title='Tendência',
    width=1200,
    height=650
)


#______________sazonalidade
import plotly.graph_objects as go

fig_sazonalidade = go.Figure()

fig_sazonalidade.add_trace(
    go.Scatter(x=sazonalidade.index, y=sazonalidade, mode='lines')
)

fig_sazonalidade.update_layout(
    title='Sazonalidade',
    xaxis_title='DthAtualizaCadastralEmpreend',
    yaxis_title='NumCPFCNPJ',
    height=500,
    width=800
)

#______________residuo
import plotly.express as px

fig_residuo = px.line(x=ts_clean.index, y=residuo)
fig_residuo.update_layout(
    xaxis_title='DthAtualizaCadastralEmpreend',
    yaxis_title='NumCPFCNPJ',
    title='Resíduo',
    width=800,
    height=400
)



#______________testes ADF e KPSS (resulados em listas)

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import plotly.graph_objs as go

# Realizando o teste ADF
resultado_adf = adfuller(ts_clean, autolag='AIC')
adf_dict = {'Estatística ADF': resultado_adf[0],
           'Número de atrasos': resultado_adf[2],
           'Valor p': resultado_adf[1]}
adf_dict = pd.DataFrame(adf_dict.items(), columns=['Teste', 'Resultado'])

# Realizando o teste KPSS
resultado_kpss = kpss(ts_clean, nlags='auto', regression='c')
kpss_dict = {'Estatística KPSS': resultado_kpss[0],
             'Número de atrasos': resultado_kpss[2],
             'Valor p': resultado_kpss[1]}
kpss_dict = pd.DataFrame(kpss_dict.items(), columns=['Teste', 'Resultado'])

# Criando a tabela com os resultados dos testes
resultado_adf_kpss = go.Figure(data=[go.Table(
    header=dict(values=list(['Teste', 'Resultado']),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[adf_dict['Teste'].tolist() + kpss_dict['Teste'].tolist(),
                       adf_dict['Resultado'].tolist() + kpss_dict['Resultado'].tolist()],
               fill_color='lavender',
               align='left'))
])

# Configurando o layout da tabela
resultado_adf_kpss.update_layout(title='Testes de Estacionariedade',
                  title_x=0.5,
                  width=600,
                  height=400)



#______________ACF
from statsmodels.graphics.tsaplots import plot_acf

fig_acf = plot_acf(ts['NumCPFCNPJ'], lags=40)
plt.xlabel('Lag')
plt.ylabel('ACF')


#import plotly.graph_objs as go
#from plotly.subplots import make_subplots
#from statsmodels.graphics.tsaplots import plot_acf
#import matplotlib.pyplot as plt

# gerar a figura com matplotlib
#fig_acf = plot_acf(ts['NumCPFCNPJ'], lags=40)

# converter a figura matplotlib para plotly
#fig_acf = make_subplots(rows=1, cols=1)
#for data in fig_acf.data:
#    fig.add_trace(go.Scatter(x=data.x, y=data.y, name=data.name))
#fig_acf.update_layout(fig_acf.layout)
#plt.show()

#______________PACF
fig_pacf = plot_acf(ts['NumCPFCNPJ'], lags=10, alpha=0.05)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.xlim([-1, 10])
#plt.show()


############################################################ GRÁFICO 1.1 ##############################################################################


import matplotlib.pyplot as plt
# agrupa os dados
counts = df_clean.groupby(['Ano_Mes', 'SigUF','SigTipoConsumidor', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()
# seria bom adicionar uma linha com a taxa de crescimento anual por Tipo de Consumidor

# filtra SigUFs 
df_clean = df_clean[df_clean['SigUF'].isin(['BA', 'AL', 'AC', 'AM', 'AP'])]
# Cria uma lista com as opções de SigUF para dropdown
sigufs = df_clean['SigUF'].unique()


fig_3 = px.bar(counts, x="Ano_Mes", y="NumCPFCNPJ", color='SigTipoConsumidor')

fig_3.update_layout(
    width=1200,
    height=600,
    xaxis_title='Ano_Mes',
    xaxis_title_font_size=16,
    yaxis_title='Novos usuários',
    yaxis_title_font_size=16,
    title='Novos usuários por tipo de consumo',
    title_font_size=20,
    title_x=.5)

############################################################ GRÁFICO 1.2 ##############################################################################

counts = df_clean.groupby(['Ano_Mes', 'SigUF','SigTipoConsumidor', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()
counts_group = counts.groupby(['Ano_Mes','SigUF']).count().reset_index().sort_values(by='Ano_Mes')
counts_group['SigUF'].unique()


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gerar_grafico_Novos_estado(uf):
    counts_pivot = counts_group[counts_group['SigUF'] == uf].pivot(
    index='Ano_Mes', columns=['SigUF'], values='NumCPFCNPJ').fillna(0)
    counts_pivot.index = pd.to_datetime(counts_pivot.index)
    counts_pivot = counts_pivot.sort_index()
    return counts_pivot

# Criar subplots
fig_nusuarios_uf = make_subplots(rows=4, cols=2, subplot_titles=['AL', 'BA', 'AP', 'AM', 'AC', 'SP', 'PA'])
# Adicionar gráficos a cada subplot
fig_nusuarios_uf.add_trace(
    go.Scatter(x=gerar_grafico_Novos_estado('AL').index, y=gerar_grafico_Novos_estado('AL').values.flatten(), mode='lines'),
    row=1, col=1)

fig_nusuarios_uf.add_trace(
    go.Scatter(x=gerar_grafico_Novos_estado('BA').index, y=gerar_grafico_Novos_estado('BA').values.flatten(), mode='lines'),
    row=1, col=2)

fig_nusuarios_uf.add_trace(
    go.Scatter(x=gerar_grafico_Novos_estado('AP').index, y=gerar_grafico_Novos_estado('AP').values.flatten(), mode='lines'),
    row=2, col=1)

fig_nusuarios_uf.add_trace(
    go.Scatter(x=gerar_grafico_Novos_estado('AM').index, y=gerar_grafico_Novos_estado('AM').values.flatten(), mode='lines'),
    row=2, col=2)

fig_nusuarios_uf.add_trace(
    go.Scatter(x=gerar_grafico_Novos_estado('AC').index, y=gerar_grafico_Novos_estado('AC').values.flatten(), mode='lines'),
    row=3, col=1)

fig_nusuarios_uf.add_trace(
    go.Scatter(x=gerar_grafico_Novos_estado('SP').index, y=gerar_grafico_Novos_estado('SP').values.flatten(), mode='lines'),
    row=3, col=2)

fig_nusuarios_uf.add_trace(
    go.Scatter(x=gerar_grafico_Novos_estado('PA').index, y=gerar_grafico_Novos_estado('PA').values.flatten(), mode='lines'),
    row=4, col=1)
# Ajustar layout
fig_nusuarios_uf.update_layout(height=800, width=1000, title_text="Novos Usuários por UF")


############################################################ GRÁFICO 2.1 ##############################################################################


# O gráfico 2 mostra a evolução temporal do Total Mensal de Empreendimentos por Estado e por Classe de Consumo;



counts2 = df_clean.groupby(['Ano', 'SigUF', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()
tabela = counts2.groupby(['Ano', 'SigUF'])['NumCPFCNPJ'].sum().reset_index()

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
    width=1500,
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


############################################################ PROPHET ##############################################################################



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
    counts = df_clean.groupby(['Ano_Mes', 'SigUF','SigTipoConsumidor', 'DscClasseConsumo']).agg({'NumCPFCNPJ': 'nunique'}).reset_index()
    counts_ = counts[counts['SigUF'] == selected_option]
    fig_3 = px.bar(counts_, x="Ano_Mes", y="NumCPFCNPJ", color="DscClasseConsumo", title="Novos usuários da Energia distribuida")
    fig_3.update_layout(
        width=1600,
        height=600,
        xaxis_title='Ano_Mes',
        xaxis_title_font_size=16,
        yaxis_title='Novos usuários',
        yaxis_title_font_size=16,
        title='Novos usuários por tipo de consumo',
        title_font_size=20,
        title_x=.5)

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

    # Introdução ao tema
    html.Div(children=[    html.Div(        children=[            html.H2('O mercado de energia distribuída no Brasil'),            
                                                                    html.P('O mercado de energia distribuída no Brasil tem se expandido graças a incentivos governamentais e à redução dos custos dos sistemas de geração distribuída, como os sistemas de energia solar fotovoltaica. A regulamentação da ANEEL (Resolução Normativa 482/2012) estabelece as regras para a conexão desses sistemas à rede elétrica e a forma de compensação pela energia gerada pelos consumidores.'),            
                                                                    html.P('A Energisa é uma empresa que oferece soluções operacionais completas para sistemas de geração distribuída, incluindo a geração, instalação e manutenção de sistemas de energia solar fotovoltaica. Além disso, a empresa oferece consultorias especializadas e linhas de financiamento específicas para a instalação desses sistemas, o que contribui para a expansão do mercado de energia distribuída no país.'),            
                                                                    html.P('Com o crescimento da demanda por sistemas de geração distribuída, a Energisa precisa prever sua expansão comercial e operacional com base no aumento do número de usuários do sistema distribuído.')        ],
            style={
                'width': '50%',
                'float': 'left',
                'padding': '10px',
                'box-sizing': 'border-box',
                'border': '1px solid #ccc',
                'text-align': 'center',
                'font-family': 'Roboto, Arial, sans-serif',
            }
        ),
        html.Div(
            children=[             html.H2('Boas práticas do projeto'),            
                                    html.P('1- todas as versões foram commitadas no git'),
                                    html.P('2- todos os scripts operam no ambiente aneel_energisa_3.8'),
                                    html.P('3- Requirements.txt atualizados diariamenteapós cada etapa do projeto'),
                                    html.P('4- A versão só é finalizada com códigos comentados')                                    
                                    ],
            style={
                'width': '50%',
                'float': 'right',
                'padding': '10px',
                'box-sizing': 'border-box',
                'border': '1px solid #ccc',
                'text-align': 'center',
                'font-family': 'Roboto, Arial, sans-serif',
            }
        )
    ],
    style={
        'width': '100%',
        'margin-top': '20px',
        'display': 'flex',
        'flex-direction': 'row',
        'justify-content': 'space-between',
        'font-family': 'Roboto, Arial, sans-serif',
    }),



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
                    html.H2('Observações do gráfico',
                        style={'font-size': '20px','padding-left': '10px','font-size': '23px', 'text-align': 'center','padding-right': '20px'}),
                    html.Div(children=[
                            'A adesão à geração distribuída de energia elétrica tem sido predominantemente por usuários em pessoa física, especialmente em residências e comércios. Isso se deve principalmente à baixa barreira de investimento, evolução das linhas de crédito e benefícios financeiros oferecidos pelo governo para incentivar o uso de fontes renováveis de energia.',
                             html.Br(), 'Embora possam ocorrer variações sazonais no número de novos usuários da energia distribuída, a tendência geral tem sido de crescimento contínuo. No entanto, observa-se uma queda no terceiro trimestre de 2022, fator relacionado à instabilidade política no país.'],
                            style={'font-size': '15px','padding-left': '10px','height': '300px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'center','padding-right': '20px'})
                ],
                style={'margin-top': '-50px', 'width': '20%', 'display': 'inline-block', 'font-family': 'Roboto, Arial, sans-serif', 'font-size': '30px', 'font-weight': '700', 'line-height': '1.2', 'color': '#181818', 'margin-top': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    ]),


    html.H2('',
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '25px', 'font-weight': '700', 'line-height': '1.2', 'color': '#0774B4', 'margin-top': '20px'}
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='grafico11',
                    figure=fig_nusuarios_uf,
                    style={'width': '50%', 'display': 'inline-block','margin': 'auto'}),
            dcc.Graph(id='grafico11',
                    figure=fig_tendencia,
                    style={'width': '50%', 'display': 'inline-block','margin': 'auto'}),
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
                    style={'width': '100%', 'display': 'inline-block','margin': 'auto'}),
            html.Div(
                id='caixa-de-texto2',
                children=[
                    html.P('',
                        style={'font-size': '20px','padding-left': '10px','font-size': '23px', 'text-align': 'center','padding-right': '20px'}),
                    html.Div('',
                            style={'font-size': '15px','padding-left': '10px','height': '300px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'center','padding-right': '20px'})
                ],
                style={'margin-top': '-50px', 'width': '00%', 'display': 'inline-block', 'font-family': 'Roboto, Arial, sans-serif', 'font-size': '30px', 'font-weight': '700', 'line-height': '1.2', 'color': '#181818', 'margin-top': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    ]),





    html.Hr(style={'margin': '10px 0', 'border-style': 'dashed', 'border-width': '1px', 'opacity': '0.5'}),

    html.H2('Estudo dos dados',
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '25px', 'font-weight': '700', 'line-height': '1.2', 'color': '#0774B4', 'margin-top': '20px'}
    ),

    html.Div([
        html.Div([
            dcc.Graph(id='grafico3',
                    figure=fig_prophet,
                    style={'width': '70%', 'display': 'inline-block'}),
            html.Div(
                id='caixa-de-texto4',
                children=[
                    html.P('Observação apartir dos dados',
                        style={'font-size': '20px','padding-left': '10px','font-size': '23px', 'text-align': 'center','padding-right': '20px'}),
                    html.Div('        - MSE: 111.00 - MAE: 6.73: - R2 Score: 0.77:  ',
                            style={'font-size': '18px','padding-left': '10px','height': '300px', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'center','padding-right': '20px'})
                ],
                style={'margin-top': '-50px', 'width': '30%', 'display': 'inline-block', 'font-family': 'Roboto, Arial, sans-serif', 'font-size': '30px', 'font-weight': '700', 'line-height': '1.2', 'color': '#181818', 'margin-top': '20px'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    ]),



    html.Hr(style={'margin': '10px 0', 'border-style': 'dashed', 'border-width': '1px', 'opacity': '0.5'}),


    html.H2('Análise macro de novos usuários da Energia distribuida',
        style={'padding-left': '30px','font-family': 'Roboto, Arial, sans-serif', 'font-size': '25px', 'font-weight': '700', 'line-height': '1.2', 'color': '#0774B4', 'margin-top': '20px'}
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='grafico22',
                    figure=fig_tendencia,
                    style={'width': '80%', 'display': 'inline-block'}),
            html.Div(
                id='caixa-de-texto22',
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




])


# http://127.0.0.1:8050/

if __name__ == '__main__':
    app.run_server(debug=True)