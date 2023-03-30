![Logo_Grupo_Energisa](https://user-images.githubusercontent.com/84819715/227146519-476224f4-6516-47bd-ba03-c4904c49e601.png)



# Energisa_aneel

O projeto tem o objetivo principal de realizar ETL de dados de Relação de empreendimentos de Geração Distribuída pela API (Aneel), realizar EDA e desenvolver uma plataforma em https(localhost) para visualizar gráficos.

## Exposição de dados - Visualização -> (...)
## Arquivo principal - Codigos -> app.ipynb
## Arquivo de testes - Codigos -> test1.ipynb

## Boas práticas:
    - Códigos comentados; 
    - Operação no ambiente vistual aneel_energisa disponibilizado em Codigos;
    - Commit no git.


## 1º etapa do projeto - (desafio energisa, http://127.0.0.1:8050/ )
### 1.1 refinamento do script (http://127.0.0.1:8040/ ))
  - ETL (Extract Transform and Load) dados da API Relação de empreendimentos de Geração Distribuída   ✔
  - EDA (Exploratory Data Analysis) em python   ✔
  - Visualização em https localhost   ✔


## 2º etapa do projeto - AWS
  - Requisição de API por API-Gateway
  - ETL por Lambda
  - Armazenamento temporário em S3
  - Armazenamento de banco em RDS
  - Evoluir a visualização de daos 


 ## 3º etapa do projeto - Inteligencia de mercado
  - Utilizar a estrutura criada para captar dados de diversas API do setor (ONS, MLE, PEE)
  - Aplicar modelos de regressão e ML
  - Evoluir a visualização de dados
