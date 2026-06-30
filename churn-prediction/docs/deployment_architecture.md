# Arquitetura de Deploy — Previsão de Churn

## 1. Decisão: Batch vs. Real-time

### Opções consideradas

| Critério | Batch (mensal) | Real-time (API síncrona) |
|---|---|---|
| Alinhamento com o caso de uso | Alto — ação de retenção é planejada, não instantânea | Baixo — não há gatilho de negócio que exija resposta em milissegundos |
| Custo de infraestrutura | Baixo — roda 1x/mês, pode ser desligado entre execuções | Mais alto — serviço precisa estar sempre disponível |
| Complexidade operacional | Baixa — job agendado, sem necessidade de SLA de latência rígido | Maior — exige monitoramento de disponibilidade contínuo |
| Velocidade de reação a mudanças de perfil do cliente | Baixa (até 30 dias de defasagem) | Alta (resposta imediata a qualquer alteração de dados) |
| Adequação ao volume de dados (dataset tabular, sem streaming) | Alta | Baixa (over-engineering para este volume) |

### Decisão tomada

**Arquitetura híbrida**, com dois modos de uso da mesma API:

1. **Modo primário — Batch mensal:** um job agendado (ex.: cron, Airflow,
   AWS Lambda + EventBridge) chama a API `/predict` em lote para toda a
   base de clientes ativos, uma vez por ciclo de faturamento. O resultado
   alimenta um relatório/dashboard para o time de retenção priorizar
   contatos no mês.
2. **Modo secundário — Consulta pontual via API:** a mesma API FastAPI
   fica disponível para consultas ad-hoc (ex.: um atendente verificando o
   risco de um cliente específico durante uma ligação de suporte), sem
   necessidade de esperar o próximo ciclo batch.

**Justificativa:** o problema de negócio (priorização de retenção) não
exige resposta em tempo real — decisões de retenção são planejadas, não
instantâneas. Mas expor o modelo via API (em vez de só um script batch)
dá flexibilidade de uso sem custo adicional relevante, e é o que o
desafio exige tecnicamente (FastAPI obrigatório).

## 2. Componentes da arquitetura

```
                          ┌─────────────────────┐
                          │   Dados de clientes   │
                          │  (CRM / Data 
warehouse) │
                          └──────────┬───────────┘
                                     │
                     ┌───────────────┴────────────────┐
                     │                                │
            ┌────────▼─────────┐           ┌──────────▼──────────┐
            │  Job batch mensal │           │  Consulta pontual    │
            │  (orquestrador)   │           │  (atendente / CRM)   │
            └────────┬─────────┘           └──────────┬──────────┘
                     │                                │
                     └───────────────┬────────────────┘
                                     │  HTTP POST /predict
                          ┌──────────▼───────────┐
                          │   API FastAPI          │
                          │   - /predict           │
                          │   - /health            │
                          │   (pipeline + MLP       │
                          │    carregados em        │
                          │    memória)             │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Artefatos do modelo   │
                          │  (models/*.joblib,     │
                          │   *.pt, metadata.json) │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  MLflow Tracking       │
                          │  (histórico de         │
                          │   experimentos)        │
                          └────────────────────────┘
```

## 3. Empacotamento e infraestrutura (opcional — bônus de nuvem)

Para o deploy opcional em nuvem (AWS/Azure/GCP), a recomendação é:

1. **Containerização**: empacotar a API FastAPI (+ artefatos do modelo)
   em uma imagem Docker, usando `uvicorn`/`gunicorn` como servidor ASGI.
2. **Hospedagem leve e barata** (adequada ao volume de tráfego deste
   projeto):
   - AWS: **App Runner** ou **ECS Fargate** (sem necessidade de gerenciar
     EC2 diretamente).
   - Azure: **Container Apps**.
   - GCP: **Cloud Run** (escala a zero quando não há tráfego — ideal
     para um caso de uso de baixo volume como este).
3. **Registro de artefatos do modelo**: os arquivos em `models/` podem
   ser versionados junto à imagem Docker (mais simples, adequado ao
   tamanho pequeno dos artefatos atuais) ou movidos para um bucket de
   objeto (S3/GCS/Blob Storage) e baixados no boot do container, caso o
   time queira atualizar o modelo sem rebuildar a imagem.
4. **Variáveis de ambiente**: `MLFLOW_TRACKING_URI`, porta, e (se
   aplicável) credenciais de storage, configuradas via secrets do
   provedor de nuvem — nunca hardcoded.

## 4. Por que não streaming em tempo real (ex.: Kafka)?

O dataset e o caso de uso são inerentemente **batch/tabular** — não há
um fluxo contínuo de eventos de telemetria do cliente que justificasse
uma arquitetura de streaming (Kafka, Kinesis, etc.). Introduzir essa
complexidade aqui seria over-engineering: aumentaria custo operacional
sem ganho de negócio correspondente, dado que a janela de decisão
(retenção) é mensal, não em milissegundos.
