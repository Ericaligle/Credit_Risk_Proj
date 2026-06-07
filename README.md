# Projeto de Modelagem de Risco de Crédito

Este repositório contém o desenvolvimento de um pipeline preditivo de ponta a ponta para a classificação de risco de crédito, determinando a probabilidade de incumprimento (*default*) de clientes. O projeto foi desenvolvido como parte de uma avaliação acadêmica de Data Science, priorizando a integridade científica, a reprodutibilidade e a governança de dados.

## Objetivo
O objetivo principal é mapear se um proponente a crédito entrará em incumprimento financeiro (`loan_status`), balanceando a mitigação de riscos com a saúde da carteira de crédito da instituição.

## Estrutura do Projeto
A estrutura do diretório segue boas práticas de organização de projetos de Ciência de Dados:

```text
CREDIT_RISK_PROJ/
├── data/
│   ├── raw/               # Dados originais e partições (treino, validação, teste)
│   └── processed/         # Dados transformados pelo pipeline
├── notebooks/             # Jupyter Notebooks para exploração e prototipagem
├── reports/
│   ├── figures/           # Gráficos e visualizações gerados nas análises
│   └── Final_Report.md    # Relatório detalhado com a metodologia e conclusões
├── src/                   # Scripts fonte com as transformações e modelagem
├── .gitignore             # Arquivos e pastas ignorados pelo Git
├── README.md              # Este arquivo de documentação
└── requirements.txt       # Dependências e versões de bibliotecas utilizadas
```

# Reprodutibilidade
Para garantir a reprodutibilidade integral da análise, siga as instruções abaixo:

## 1. Requisitos
Certifique-se de ter o Python 3 instalado. Recomenda-se a criação de um ambiente virtual antes de instalar as dependências.


### Criar ambiente virtual
 
```
python -m venv .venv
```

### Ativar ambiente (Windows)
```
.venv\Scripts\activate
```

### Ativar ambiente (Linux/macOS)
```
source .venv/bin/
```

## 2. Instalação das Dependências

Instale as bibliotecas necessárias listadas no arquivo requirements.txt:

```
pip install -r requirements.txt
```

## 3. Verificação inicial dos dados

A integridade dos dados é o pilar deste projeto. Os dados brutos foram obtidos a partir do [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). O dataset original possui **32.581 registros (linhas)** e **12 colunas**, que são processados no notebook `notebooks/0-Dataset_split.ipynb`, onde implementamos uma rotina de auditoria que gera o relatório de qualidade abaixo:

| Coluna | Num of Values | % of Missing Values | Unique Values (Cat.) | Mean (Numerical) | Python Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cb_person_cred_hist_length` | 32581 | 0.000000 | NaN | 5.804211 | int64 |
| `cb_person_default_on_file` | 32581 | 0.000000 | 2.0 | NaN | str |
| `loan_amnt` | 32581 | 0.000000 | NaN | 9589.371106 | int64 |
| `loan_grade` | 32581 | 0.000000 | 7.0 | NaN | str |
| `loan_int_rate` | 29465 | 9.563856 | NaN | 11.011695 | float64 |
| `loan_intent` | 32581 | 0.000000 | 6.0 | NaN | str |
| `loan_percent_income` | 32581 | 0.000000 | NaN | 0.170203 | float64 |
| `loan_status` | 32581 | 0.000000 | NaN | 0.218164 | int64 |
| `person_age` | 32581 | 0.000000 | NaN | 27.734600 | int64 |
| `person_emp_length` | 31686 | 2.747000 | NaN | 4.789686 | float64 |
| `person_home_ownership` | 32581 | 0.000000 | 4.0 | NaN | str |
| `person_income` | 32581 | 0.000000 | NaN | 66074.848470 | int64 |


## 4. Descrição das Variáveis

| Nome da Variável | Grupo | Descrição |
| :--- | :--- | :--- |
| `person_age` | Pessoal | Idade do solicitante |
| `person_income` | Financeiro | Renda Anual |
| `person_home_ownership` | Pessoal | Status da moradia (própria, alugada, etc.) |
| `person_emp_length` | Financeiro | Tempo de emprego atual (anos) |
| `loan_intent` | Empréstimo | Motivo do pedido |
| `loan_grade` | Empréstimo | Nível de risco do empréstimo |
| `loan_amnt` | Empréstimo | Valor solicitado |
| `loan_int_rate` | Empréstimo | Taxa de juros aplicada |
| `loan_percent_income` | Empréstimo | Razão entre valor do empréstimo e renda |
| `cb_person_default_on_file` | Crédito | Histórico de inadimplência (S/N) |
| `cb_person_cred_hist_length` | Crédito | Tempo de histórico de crédito (anos) |
| **`loan_status`** | **Alvo** | **Status: 0 (Pago) / 1 (Default)** |


---

## 5. Execução da Análise
O projeto foi estruturado para permitir que o ciclo de vida analítico — desde a divisão estratificada dos dados até a avaliação final no conjunto de teste holdout — seja reproduzido de forma consistente. O script `notebooks/0-Dataset_split.ipynb` realiza a partição estanque dos dados (70% treino, 15% validação, 15% teste), utilizando a semente aleatória `RANDOM_STATE = 42` para assegurar resultados replicáveis.

**Metodologia Tratamento de Dados** : Análise de mecanismos de perda (MAR/MCAR) e imputação condicional customizada para preservar a coerência financeira.

**Pipeline**: Utilização de pipelines do scikit-learn com transformadores robustos (RobustScaler) para mitigar o impacto de outliers.

**PCA**: Investigação da redução de dimensionalidade, decidindo pela manutenção das variáveis originais em favor da interpretabilidade e transparência exigidas pelo negócio.

**Modelagem**: Regressão Logística com ajuste de pesos (balanced) como modelo campeão.