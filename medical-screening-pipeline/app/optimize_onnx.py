import os
import pandas as pd
import joblib
import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import time

def convert_to_onnx():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
    onnx_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.onnx')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'medical_tc_train.csv')
    
    print("A carregar dados para garantir ajuste perfeito do TF-IDF...")
    df = pd.read_csv(data_path).dropna(subset=['medical_abstract', 'condition_label'])
    X = df['medical_abstract'].astype(str).head(1000)  # Amostra robusta para ajuste rápido
    y = df['condition_label'].head(1000)

    print("Treinando pipeline de referência para ONNX...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
        ('clf', LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X, y)
    
    # Definir o tipo de entrada (string text tensor)
    initial_type = [('medical_abstract', StringTensorType([None, 1]))]
    
    print("Convertendo pipeline para ONNX...")
    options = {id(pipeline.named_steps['clf']): {'zipmap': False}}
    onnx_model = convert_sklearn(
        pipeline, 
        initial_types=initial_type, 
        options=options,
        target_opset=17
    )
    
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Modelo ONNX guardado com sucesso em: {onnx_path}")

    # Benchmark simples de latência
    sample_text = ["Patient presents with acute chest pain and shortness of breath."]
    
    # Latência Scikit-Learn
    start = time.time()
    for _ in range(50):
        pipeline.predict(sample_text)
    sklearn_time = (time.time() - start) / 50

    # Latência ONNX Runtime
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    start = time.time()
    for _ in range(50):
        sess.run(None, {input_name: np.array(sample_text, dtype=object).reshape(-1, 1)})
    onnx_time = (time.time() - start) / 50

    print(f"\n--- Comparativo de Latência (Média de 50 execuções) ---")
    print(f"Scikit-Learn Standard: {sklearn_time * 1000:.4f} ms")
    print(f"ONNX Runtime Optimized: {onnx_time * 1000:.4f} ms")

if __name__ == '__main__':
    convert_to_onnx()