# Purchase Intention — Previsão de Propensão de Compra

Pipeline de Machine Learning que prevê se uma sessão de navegação em um
e-commerce vai resultar em compra (`Revenue = True/False`), a partir do
comportamento do usuário durante a sessão. Projeto desenvolvido para o
Tech Challenge — Fase 2 (Machine Learning Engineering) da Pós Tech FIAP.

## Problema de negócio

O dataset **Online Shoppers Purchasing Intention** (UCI/Kaggle, 12.330
sessões) contém métricas de navegação — páginas visitadas, tempo em cada
categoria de página, taxa de rejeição, mês do ano, tipo de visitante, entre
outras — e o rótulo binário `Revenue`, indicando se a sessão terminou em
compra. O objetivo é treinar um classificador que apoie decisões de
negócio como priorização de remarketing e otimização de UX.

## Arquitetura do projeto

purchase-intention/
├── src/purchase_intention/
│ ├── config.py # Configuração tipada (dataclasses) carregada de YAML
│ ├── data/loader.py # Carregamento e validação de schema do dataset bruto
│ ├── features/preprocessing.py # ColumnTransformer (StandardScaler + OneHotEncoder)
│ ├── models/train.py # Construção do pipeline de modelo e avaliação
│ └── pipeline/
│ ├── preprocess.py # Estágio DVC: limpeza do dataset bruto
│ ├── train.py # Estágio DVC: treino + tracking + registro no MLflow
│ └── promote.py # Promoção do modelo ao alias "champion" (gate de qualidade)
├── tests/ # Testes unitários (pytest)
├── configs/config.yaml # Hiperparâmetros, caminhos e configuração do MLflow
├── data/raw/ # Dataset bruto (versionado via DVC, não Git)
├── data/processed/ # Dataset limpo, gerado pelo estágio "preprocess"
├── dvc.yaml # Pipeline reprodutível (preprocess -> train)
├── Dockerfile # Build multi-stage com Poetry
├── pyproject.toml # Dependências gerenciadas via Poetry
└── .env.example # Modelo de variáveis de ambiente


**Decisões de arquitetura:**

- **Configuração centralizada** (`configs/config.yaml`): nenhum hiperparâmetro
  ou caminho está hardcoded no código. Isso torna experimentos reprodutíveis
  e fáceis de ajustar sem editar Python.
- **Separação preprocess / train como estágios DVC distintos**: permite que o
  DVC pule etapas cujas dependências não mudaram (rebuild seletivo), e
  documenta explicitamente o fluxo de dados até o modelo.
- **MLflow com backend SQLite** (`sqlite:///mlflow.db`): o Model Registry do
  MLflow exige um backend de banco de dados — não funciona com o backend de
  arquivos padrão (`mlruns/`). SQLite foi escolhido por não exigir nenhuma
  infraestrutura adicional, mantendo o projeto simples de rodar localmente.
- **Aliases em vez de Stages no Model Registry**: as "stages" tradicionais do
  MLflow (Staging/Production/Archived) estão descontinuadas desde a versão
  2.9. Este projeto usa o mecanismo atual de **aliases** (`@champion`), que é
  a prática recomendada pela documentação oficial do MLflow.
- **Gate de qualidade automatizado na promoção** (`pipeline/promote.py`): um
  modelo só é promovido a `champion` se superar (ou empatar com) o ROC AUC
  do champion atual. Isso implementa, de forma simples, o mesmo princípio de
  governança usado em pipelines de CI/CD de ML em produção.

## Pré-requisitos

- Python 3.11 ou 3.12
- [Poetry](https://python-poetry.org/) 1.8+
- [DVC](https://dvc.org/) (instalado automaticamente via Poetry, grupo dev)
- Docker Desktop (opcional, para rodar o pipeline containerizado)

## Como rodar o projeto

### 1. Instalar dependências

```bash
poetry install
```

### 2. Reproduzir o pipeline completo (DVC)

```bash
poetry run dvc repro
```

Isso executa, em sequência:
1. **`preprocess`** — carrega `data/raw/online_shoppers_intention.csv`, remove
   duplicatas, salva `data/processed/dataset.csv`.
2. **`train`** — treina um `RandomForestClassifier` sobre os dados
   processados, registra parâmetros/métricas/modelo no MLflow, e registra a
   nova versão no Model Registry.

Rodar `dvc repro` novamente sem alterar nada não reexecuta nenhum estágio
(`Data and pipelines are up to date`) — essa é a reprodutibilidade
seletiva que o DVC garante ao rastrear hashes de dependências.

### 3. Promover o modelo (gate de qualidade)

```bash
poetry run python -m purchase_intention.pipeline.promote --config configs/config.yaml
```

Compara a versão recém-treinada com o `champion` atual (por ROC AUC) e só
promove se ela for igual ou melhor. Registra `approved_by` e
`approval_date` como tags da versão, para auditabilidade.

### 4. Consultar experimentos no MLflow UI

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra `http://127.0.0.1:5000` para visualizar runs, métricas, comparações
entre versões e o modelo atualmente com o alias `champion`.

### 5. Rodar os testes

```bash
poetry run pytest
```

### 6. Rodar via Docker

```bash
docker build -t purchase-intention .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  purchase-intention
```

O container executa `preprocess` e `train` em sequência (ver `CMD` no
`Dockerfile`). Os volumes montados garantem que o dataset bruto seja lido de
fora da imagem e que os artefatos gerados persistam no host.

## Versionamento de dados (DVC)

O dataset bruto é versionado com DVC, não com Git:

```bash
poetry run dvc add data/raw/online_shoppers_intention.csv
poetry run dvc push   # envia para o remote configurado
poetry run dvc pull   # recupera os dados em uma nova máquina/clone
```

O remote configurado neste projeto é uma pasta local (ver `.dvc/config`),
suficiente para desenvolvimento e para fins didáticos deste desafio. Em um
ambiente de produção, o mesmo fluxo funciona apontando o remote para
Amazon S3, Google Cloud Storage ou Azure Blob Storage.

## Reprodutibilidade

- **Seed fixa** (`random_state=42` em `configs/config.yaml`) aplicada ao
  Python `random`, ao NumPy e ao `train_test_split`/`RandomForestClassifier`
  do scikit-learn.
- **Dependências travadas** via `poetry.lock`.
- **Dados versionados** via DVC — cada commit do Git aponta para uma versão
  exata do dataset bruto.
- **Pipeline declarativo** via `dvc.yaml`/`dvc.lock` — qualquer pessoa que
  clone o repositório e rode `dvc repro` reproduz os mesmos resultados.

## Métricas do modelo

O estágio `train` avalia o modelo em um conjunto de teste (20% dos dados,
estratificado) e registra:

| Métrica | Descrição |
|---|---|
| Accuracy | Acurácia geral |
| Precision | Precisão da classe positiva (compra) |
| Recall | Sensibilidade da classe positiva |
| F1-score | Média harmônica entre precisão e recall |
| ROC AUC | Área sob a curva ROC — métrica principal usada no gate de promoção |

Essas métricas também são salvas em `metrics.json` na raiz do projeto,
rastreado pelo DVC (`dvc metrics show` / `dvc metrics diff`).

## Autor

Flávio José Joaquim Frederico — Pós Tech em Machine Learning Engineering
(FIAP), Fase 2.