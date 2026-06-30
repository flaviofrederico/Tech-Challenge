# Model Card — Previsão de Churn (Telco)

## 1. Visão geral

| Campo | Valor |
|---|---|
| **Nome do modelo** | `churn_mlp` (MLP em PyTorch) |
| **Versão** | 1.0.0 |
| **Tipo de tarefa** | Classificação binária (churn / não-churn) |
| **Arquitetura** | MLP feed-forward: `Input(40) -> 64 -> 32 -> 1 (logit)`, ReLU + Dropout(0.3) |
| **Framework** | PyTorch 2.x |
| **Data de treino** | Ver `models/model_metadata.json` (`run_id` no MLflow) |
| **Dataset de origem** | [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7.043 clientes, 21 colunas |
| **Responsáveis** | Grupo Tech Challenge — Fase 1 |

## 2. Uso pretendido

**Uso primário:** apoiar o time de retenção/CRM de uma operadora de
telecomunicações a **priorizar contatos proativos** com clientes de alto
risco de cancelamento, dentro de um ciclo mensal de faturamento.

**Fora do escopo (não usar para):**
- Decisões automatizadas que afetem o cliente sem revisão humana (ex.:
  negar serviço, alterar preço unilateralmente).
- Qualquer decisão de crédito, elegibilidade ou similar — o modelo **não**
  foi treinado nem validado para esse propósito.
- Generalização para outras operadoras, países ou períodos sem
  re-treinamento e re-validação (o dataset é de uma única empresa,
  período e mercado específicos, possivelmente já desatualizado).

## 3. Dados de treinamento

- **Fonte:** dataset público IBM/Kaggle, adaptação de uma base de
  amostra da IBM Cognos Analytics.
- **Tamanho:** 7.043 registros, 19 features preditivas (excluindo ID e
  target), split 72% treino / 8% validação / 20% teste (estratificado).
- **Limpeza aplicada:** 11 registros (0,16%) com `TotalCharges` em branco
  — todos correspondentes a clientes com `tenure=0` — foram preenchidos
  com `0.0`. Nenhum registro foi descartado.
- **Desbalanceamento:** 26,5% de churn / 73,5% de não-churn (moderado,
  tratado via `pos_weight` na loss, sem oversampling).

## 4. Métricas de performance (conjunto de teste hold-out, n=1.409)

| Métrica | Valor (threshold padrão 0.5) |
|---|---|
| ROC-AUC | 0.843 |
| PR-AUC | 0.639 |
| F1-Score | 0.628 |
| Precision | 0.499 |
| Recall | 0.845 |
| Accuracy | 0.734 |
| Custo de negócio estimado | R$ 55.910 |
| Custo de negócio (threshold ótimo = 0.10) | R$ 24.200 |

**Comparação com baselines (mesmo conjunto de teste):**

| Modelo | ROC-AUC | F1 | Recall | Custo de negócio |
|---|---|---|---|---|
| Dummy (most_frequent) | 0.500 | 0.000 | 0.000 | R$ 299.200 |
| Regressão Logística | 0.841 | 0.614 | 0.783 | R$ 73.440 |
| Random Forest | 0.842 | 0.624 | 0.759 | R$ 79.560 |
| **MLP (PyTorch)** | **0.843** | **0.628** | **0.845** | **R$ 55.910** |

**Interpretação:** os 4 modelos não-triviais convergem para ROC-AUC muito
próximo (~0,84) — esperado, já que os sinais dominantes do dataset
(tipo de contrato, tempo de relacionamento, tipo de internet) têm relação
majoritariamente monotônica/linear com o churn, reduzindo a vantagem
estrutural de uma rede neural sobre modelos mais simples. O MLP se
diferencia principalmente por um **recall mais alto** e um **custo de
negócio menor**, o critério que efetivamente importa para a diretoria.

## 5. Métrica de negócio e premissas de custo

Custo esperado = `FP × R$30 (ação de retenção) + FN × R$800 (cliente perdido)`.

**Premissas documentadas (ajustar com o time de negócio antes de produção real):**
- `R$30`: custo aproximado de uma ação de retenção (desconto pontual,
  ligação do time de CRM) — **estimativa ilustrativa**, não validada com
  dados financeiros reais da operadora.
- `R$800`: aproximação de ~12 meses de `MonthlyCharges` médio (R$64,76 ×
  12 ≈ R$777) como proxy do valor perdido por churn — também uma
  simplificação; não contabila custo de aquisição de um substituto, nem
  o valor de vida útil (LTV) real do cliente.

Essas duas constantes (`COST_RETENTION_ACTION`, `COST_CHURN_LOSS`) estão
centralizadas em `src/churn_prediction/config.py` e devem ser recalibradas
com dados financeiros reais antes de qualquer uso em produção.

## 6. Análise de subgrupos (fairness / viés)

Performance do modelo (threshold 0.5) por subgrupo, no conjunto de teste:

| Subgrupo | n | ROC-AUC | Recall | Precision | Churn rate real |
|---|---|---|---|---|---|
| Gênero: Male | 722 | 0.848 | 0.867 | 0.480 | 25,1% |
| Gênero: Female | 687 | 0.839 | 0.824 | 0.520 | 28,1% |
| Não idoso | 1.187 | 0.844 | 0.808 | 0.468 | 23,3% |
| **Idoso (SeniorCitizen=1)** | 222 | **0.781** | 0.949 | 0.596 | **44,1%** |
| Contrato Two year | 336 | 0.761 | 0.000* | — | 2,7% |
| Contrato Month-to-month | 773 | 0.748 | 0.933 | — | 42,6% |
| Contrato One year | 300 | 0.746 | 0.250 | — | 12,0% |

\* *Recall 0.000 em "Two year" não indica falha grave: há apenas ~9
churners reais nesse subgrupo no teste, e a baixíssima taxa de base
torna qualquer métrica de recall instável com tamanho de amostra
pequeno — não interpretar como o modelo "ignorando" esse segmento sem
olhar o volume absoluto.*

**Achados relevantes:**

1. **Gênero**: diferença pequena entre Male/Female (ROC-AUC 0.848 vs
   0.839) — não há indício de viés relevante por essa variável.
2. **Idade (SeniorCitizen)**: o modelo tem ROC-AUC visivelmente mais
   baixo para clientes idosos (0.781 vs 0.844), embora o recall seja
   mais alto nesse grupo (population com churn rate real quase 2x maior).
   Isso sugere que o modelo é **menos discriminativo** dentro do
   subgrupo de idosos, possivelmente por amostra menor (n=222 no teste)
   e/ou por esse grupo ter um perfil de risco mais homogêneo
   (alto risco quase generalizado). **Recomendação:** se o modelo for
   usar para decisões com impacto direto a esse subgrupo, considerar
   monitoramento segmentado contínuo e, se a diferença persistir,
   avaliar re-balanceamento ou um modelo específico para o segmento.
3. **Tipo de contrato**: o ROC-AUC "dentro do subgrupo" cai bastante
   (0.75 vs 0.84 global) porque, ao condicionar em `Contract`, removemos
   a própria variável mais discriminativa do modelo — isso é esperado
   estatisticamente e não indica, por si só, um problema de viés.

## 7. Limitações conhecidas

- **Dataset estático e potencialmente desatualizado**: não há
  informação temporal/sazonal; o comportamento de churn pode ter mudado
  desde a coleta original dos dados.
- **Sem dados de atendimento ao cliente, NPS ou histórico de
  reclamações**: sinais que tipicamente são fortes preditores de churn
  em operadoras reais e não estão presentes neste dataset.
- **Amostra única de uma operadora**: o modelo não deve ser usado
  diretamente em outra empresa/mercado sem re-treinamento.
- **Calibração de probabilidade não validada**: as métricas reportadas
  (ROC-AUC, F1, etc.) avaliam ranking e classificação, não a calibração
  exata da probabilidade (ex.: "70% de probabilidade" pode não
  corresponder literalmente a 70% de frequência empírica). Se a
  probabilidade for usada para cálculos financeiros diretos (ex.:
  EV = proba × valor), recomenda-se validar calibração (reliability
  diagram) antes do uso.
- **Threshold ótimo (0.10) é sensível às premissas de custo da Seção 5**
  — se os custos reais de retenção/perda forem diferentes dos assumidos,
  o threshold deve ser recalculado.

## 8. Cenários de falha conhecidos

| Cenário | Risco | Mitigação recomendada |
|---|---|---|
| Cliente novo (tenure baixo) sem histórico de pagamento | Maior incerteza na predição (TotalCharges pouco informativo) | Monitorar performance segmentada por faixa de tenure |
| Mudança de mix de produtos da operadora (ex.: novo plano não presente no treino) | Modelo não generaliza para categorias nunca vistas | `OneHotEncoder(handle_unknown="ignore")` evita erro, mas a predição pode ser pouco confiável; re-treinar ao introduzir novos planos |
| Drift de custo de negócio (inflação, mudança de política de retenção) | Threshold ótimo (0.10) deixa de refletir o custo real | Revisitar `COST_RETENTION_ACTION` / `COST_CHURN_LOSS` periodicamente (ver plano de monitoramento) |
| Uso do modelo fora do ciclo mensal pretendido (ex.: decisão em tempo real durante uma ligação) | Modelo não foi validado para esse padrão de uso | Restringir uso ao batch mensal documentado na arquitetura de deploy |

## 9. Reprodutibilidade

- Seed fixada: `RANDOM_SEED = 42` (NumPy, PyTorch, splits do scikit-learn).
- Hiperparâmetros completos: ver `src/churn_prediction/config.py::MLPConfig`.
- Experimento rastreado no MLflow: `experiment_name="churn-prediction"`,
  `run_id` salvo em `models/model_metadata.json`.
- Pipeline de pré-processamento serializado em
  `models/preprocessing_pipeline.joblib`, garantindo que a mesma
  transformação usada em treino é aplicada em produção (sem skew).

## 10. Contato e manutenção

Este modelo foi desenvolvido como parte de um desafio educacional
(Tech Challenge — Fase 1). Para uso em produção real, recomenda-se:
revalidação das premissas de custo de negócio (Seção 5) com o time
financeiro, ampliação da análise de subgrupos (Seção 6) com dados de
um período mais longo, e a configuração de monitoramento contínuo
descrita em `docs/monitoring_plan.md`.
