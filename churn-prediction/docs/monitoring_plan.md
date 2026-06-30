# Plano de Monitoramento — Modelo de Previsão de Churn

## 1. Objetivo

Garantir que o modelo em produção continue performando dentro do
esperado, detectar degradação (drift de dados ou de performance) antes
que ela impacte as decisões de negócio, e definir um playbook claro de
resposta quando algo sair do previsto.

## 2. Métricas a monitorar

### 2.1 Métricas operacionais (infraestrutura/API)

| Métrica | Como medir | Alvo (SLO) |
|---|---|---|
| Latência p95 do `/predict` | Header `X-Process-Time-Ms` (já implementado no middleware) | < 200ms |
| Taxa de erro (HTTP 5xx) | Logs estruturados da API (`request_completed`) | < 1% das requisições |
| Disponibilidade do `/health` | Health check periódico (ex.: cada 1 min) | ≥ 99% (ambiente de portfólio/demo) |

### 2.2 Métricas de qualidade de dados (data drift)

| Métrica | Como medir | Gatilho de alerta |
|---|---|---|
| Distribuição de features numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) | Comparar histograma/estatísticas (média, desvio) do batch atual vs. baseline de treino (ex.: teste KS — Kolmogorov-Smirnov) | p-valor < 0.05 em 2+ execuções consecutivas |
| Distribuição de categorias (`Contract`, `InternetService`, etc.) | Comparar proporções vs. baseline (ex.: divergência de Jensen-Shannon ou Chi-quadrado) | Mudança > 10 p.p. em qualquer categoria principal |
| Taxa de valores fora do domínio esperado (schema) | Validação Pandera (`API_INPUT_SCHEMA`) a cada requisição | Qualquer falha de schema em produção (alerta imediato) |
| Volume de predições por dia/mês | Contagem de chamadas ao `/predict` | Queda > 50% vs. média móvel (pode indicar problema de integração upstream) |

### 2.3 Métricas de qualidade do modelo (performance drift)

> **Importante:** o rótulo real (`Churn`) só é conhecido com atraso (o
> cliente de fato cancela ou não, ao longo do mês seguinte). Por isso,
> essas métricas são calculadas em uma **janela retroativa** (ex.:
> comparando as predições de 30-60 dias atrás contra o resultado real
> observado agora), não em tempo real.

| Métrica | Frequência | Gatilho de alerta |
|---|---|---|
| ROC-AUC realizado (retroativo) | Mensal | Queda > 0.05 em relação à baseline de teste (0.843) |
| Recall realizado | Mensal | Queda > 0.08 em relação à baseline (0.845) — prioridade alta, pois recall baixo significa deixar clientes de risco passarem |
| Custo de negócio realizado | Mensal | Aumento > 20% em relação ao custo esperado em teste |
| Performance por subgrupo (idosos, tipo de contrato) | Trimestral | Ver Seção 6 do Model Card — alerta se a diferença de ROC-AUC entre subgrupos aumentar |

## 3. Ferramentas sugeridas

- **Logs estruturados (já implementados)**: toda requisição da API e
  todo evento de treino são logados em JSON (`churn_prediction.logging_config`),
  prontos para ingestão em uma stack de observabilidade (ex.: CloudWatch
  Logs, Grafana Loki, ELK).
- **MLflow**: histórico de todos os experimentos de treino/re-treino,
  permitindo comparar a performance de cada versão do modelo ao longo
  do tempo.
- **Dashboards**: um painel simples (Grafana, ou até uma planilha
  automatizada) cruzando as métricas das Seções 2.1–2.3 mês a mês.

## 4. Playbook de resposta a alertas

| Alerta disparado | Primeira ação | Ação de correção |
|---|---|---|
| Falha de schema em produção | Verificar se há uma integração upstream gerando dados malformados (ex.: novo valor categórico não visto em treino) | Ajustar `API_INPUT_SCHEMA` ou corrigir a fonte de dados; nunca silenciar o erro sem investigar |
| Drift de distribuição de features | Confirmar se é uma mudança real de comportamento do cliente ou um problema de pipeline de dados upstream | Se for mudança real e persistente: agendar re-treinamento com dados recentes |
| Queda de ROC-AUC/Recall realizado | Levantar uma amostra de casos recentes (erros do modelo) para inspeção manual | Re-treinar com dados mais recentes; se a queda for abrupta, investigar se houve mudança de produto/política comercial não capturada no modelo |
| Aumento do custo de negócio realizado | Verificar se as premissas de custo (`COST_RETENTION_ACTION`, `COST_CHURN_LOSS`) ainda são válidas | Recalibrar as constantes de custo com o time financeiro e recalcular o threshold ótimo |
| Latência acima do SLO | Verificar carga/recursos do container (CPU/memória) | Escalar verticalmente/horizontalmente; revisar se o pré-processamento pode ser otimizado |

## 5. Cadência de re-treinamento

- **Padrão:** re-treinamento trimestral com os dados acumulados mais
  recentes, mesmo sem alertas — para capturar mudanças graduais de
  comportamento do cliente.
- **Gatilho de re-treinamento antecipado:** qualquer alerta da Seção 2.3
  (queda de ROC-AUC/Recall ou aumento de custo) confirmado após
  investigação manual.
- **Processo:** todo re-treinamento deve gerar um novo `run` no MLflow,
  ser comparado contra a versão em produção usando o mesmo conjunto de
  teste hold-out (ou um hold-out mais recente), e só substituir o modelo
  em produção se igualar ou superar a performance atual nas métricas
  de negócio (não apenas nas técnicas).
