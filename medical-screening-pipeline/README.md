# Medical Screening Pipeline - Tech Challenge (Fase 3)

Pipeline MLOps completo de triagem automática de laudos médicos baseado em Processamento de Linguagem Natural (NLP), FastAPI, Prometheus, Grafana, Airflow e ONNX Runtime.

## 🏛️ 1. Decisão Arquitetural em Nuvem (AWS)
Para o deploy em produção deste sistema de triagem de urgência hospitalar, optou-se por uma arquitetura baseada em **AWS (Amazon Web Services)** utilizando **AWS ECS (Elastic Container Service) com AWS Fargate**, orquestrado via contentores Docker.
- **Justificativa (Real-time vs Batch):** A triagem clínica exige baixa latência em tempo real para tomada de decisões médicas urgentes. Uma arquitetura baseada em API REST stateless alojada no ECS Fargate garante escalabilidade automática horizontal mediante picos de procura de laudos, sem necessidade de gestão direta de servidores.
- **Armazenamento:** O dataset e os artefatos de modelos versionados são mantidos em buckets **Amazon S3**.

## 🚀 2. Como Executar o Projeto
1. **Treinar o Modelo:**
   ```bash
   python app/train.py