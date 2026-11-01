from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Carregamento do modelo no ciclo de vida (Lifespan) moderno do FastAPI
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Modelo carregado com sucesso de {MODEL_PATH}")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
    else:
        print(f"Aviso: Modelo não encontrado em {MODEL_PATH}.")
    yield

app = FastAPI(
    title="Medical Screening Pipeline API",
    description="API de triagem automática de laudos médicos com Machine Learning e Observabilidade",
    version="1.0.0",
    lifespan=lifespan
)

# Métricas Prometheus
REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total de requisições HTTP", 
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", 
    "Latência das requisições HTTP em segundos", 
    ["endpoint"]
)

class MedicalInput(BaseModel):
    medical_abstract: str

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API do Medical Screening Pipeline (Fase 3)"}

@app.post("/predict")
def predict(input_data: MedicalInput):
    global model
    start_time = time.time()
    
    # Carregamento automático de segurança caso o lifespan não tenha sido acionado (ex: em alguns contextos de TestClient)
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            pass

    if model is None:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=500).inc()
        raise HTTPException(status_code=500, detail="Modelo de Machine Learning não carregado.")
    
    try:
        prediction = model.predict([input_data.medical_abstract])
        confidence = model.predict_proba([input_data.medical_abstract]).max() if hasattr(model, "predict_proba") else None
        
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=200).inc()
        
        return {
            "condition_label": int(prediction[0]),
            "confidence": float(confidence) if confidence is not None else "N/A"
        }
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)