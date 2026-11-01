import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_model():
    # Caminho para o dataset de treino dentro da pasta data/
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'medical_tc_train.csv')
    print(f'Carregando dados de: {data_path}')
    
    df = pd.read_csv(data_path)
    
    # Removendo valores nulos com base nas colunas reais do dataset
    df = df.dropna(subset=['medical_abstract', 'condition_label'])
    
    X = df['medical_abstract'].astype(str)
    y = df['condition_label']
    
    print('Treinando pipeline TF-IDF + Logistic Regression...')
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=300))
    ])
    
    pipeline.fit(X, y)
    
    # Criando a pasta models e guardando o artefato treinado
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
    
    joblib.dump(pipeline, model_path)
    print(f'Modelo treinado e salvo com sucesso em: {model_path}')

if __name__ == '__main__':
    train_model()