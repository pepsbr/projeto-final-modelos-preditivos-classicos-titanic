# Projeto Final — Estatísticas e Pré-processamento do Titanic

Projeto acadêmico de Modelos Preditivos Clássicos sobre análise exploratória e preparação do dataset Titanic para uma tarefa de classificação da variável `Survived`.

## Análises previstas no notebook

- Inspeção e estatística descritiva dos dados.
- Codificação de `Sex` e `Embarked`.
- Tratamento de valores ausentes em `Age` por mediana de classe e sexo.
- Separação holdout em treino e teste (70%/30%).
- Padronização de `Age` e `Fare` com `StandardScaler`, ajustado somente no treino.

## Dados

O notebook utiliza o conjunto de treino do Titanic, carregado de [agconti/kaggle-titanic](https://github.com/agconti/kaggle-titanic/blob/master/data/train.csv). O CSV não está incluído; é obtido pela URL pública no código.

## Arquivos publicados

- `README.md`: descrição do projeto e instruções.
- `requirements.txt`: dependências Python.
- `.gitignore`: exclusões para ambientes locais, credenciais e dados baixados.

**O notebook `01_Equipe_04_EDA_Classificação_Titanic.ipynb` ainda não foi enviado ao repositório.** Ele está na pasta local do projeto e precisa ser adicionado para que as instruções de execução abaixo funcionem.

## Como executar

Recomendado: Python 3.10 ou superior. Após adicionar o notebook: instale as dependências com `python -m pip install -r requirements.txt`, execute `jupyter notebook` e abra o arquivo do notebook.

## Equipe

- Mauro Roberto Barbosa Cahu
- Pedro Paulo Sant'Anna Belo Ramos
- Renato Vasconcellos Bittencourt Neto
- Vinicius Dantas Batista da Silva

## Data do projeto

**07/08/2026**. A data de criação exibida pelo GitHub corresponde à criação real do repositório na plataforma.
