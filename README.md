# Projeto Final — Estatísticas e Pré-processamento do Titanic

Projeto acadêmico da disciplina de Modelos Preditivos Clássicos. O notebook realiza análise exploratória e prepara o conjunto de dados Titanic para uma futura tarefa de classificação da variável `Survived`.

## O que foi feito

- Inspeção inicial dos dados e estatísticas descritivas.
- Codificação de variáveis categóricas (`Sex` e `Embarked`).
- Remoção de colunas identificadoras e com muitos dados ausentes.
- Tratamento de valores ausentes em `Age` por mediana de classe e sexo.
- Separação em treino e teste com holdout de 70%/30%.
- Padronização das variáveis `Age` e `Fare` com `StandardScaler`, ajustado somente no conjunto de treino.

## Arquivos

- `01_Equipe_04_EDA_Classificação_Titanic.ipynb`: notebook com código, visualizações, resultados e conclusão.
- `requirements.txt`: bibliotecas necessárias para executar o notebook.
- `.gitignore`: exclusões de arquivos locais, ambientes virtuais, credenciais e dados baixados.

## Dados

O notebook lê o conjunto de treino do Titanic diretamente de [agconti/kaggle-titanic](https://github.com/agconti/kaggle-titanic/blob/master/data/train.csv). O arquivo CSV não está armazenado neste repositório; ele é obtido pela URL pública indicada no código.

## Como executar

Recomendado: Python 3.10 ou superior.

1. Clone este repositório e acesse a pasta.
2. Crie e ative um ambiente virtual Python.
3. Instale as dependências: `python -m pip install -r requirements.txt`.
4. Inicie o Jupyter: `jupyter notebook`.
5. Abra `01_Equipe_04_EDA_Classificação_Titanic.ipynb` e execute as células em ordem.

## Equipe

- Mauro Roberto Barbosa Cahu
- Pedro Paulo Sant'Anna Belo Ramos
- Renato Vasconcellos Bittencourt Neto
- Vinicius Dantas Batista da Silva

## Data do projeto

Data informada para o projeto: **07/08/2026**. A data de criação que o GitHub exibe para o repositório corresponde à criação real na plataforma.
