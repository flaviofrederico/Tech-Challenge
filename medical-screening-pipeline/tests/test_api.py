from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Bem-vindo" in response.json()["message"]

def test_predict_endpoint():
    # Teste de triagem com um resumo médico simulado
    payload = {
        "medical_abstract": "Patient presents with severe chest pain and shortness of breath, indicating a potential acute coronary syndrome."
    }
    response = client.post("/predict", json=payload)
    # Se o modelo estiver carregado, deve retornar 200 com a predição
    assert response.status_code == 200
    data = response.json()
    assert "condition_label" in data
    assert "confidence" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Verifica se o Prometheus está a expor métricas textuais
    assert "http_requests_total" in response.text