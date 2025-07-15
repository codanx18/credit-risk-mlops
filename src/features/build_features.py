import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def build_feature_pipeline(df_train):
    """Construye y entrena un pipeline de preprocesamiento de características."""
    
    categorical_features = df_train.select_dtypes(include=['object']).columns
    numerical_features = df_train.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def process_features(preprocessor, df, fit=False):
    """Aplica el pipeline de preprocesamiento a los datos."""
    if fit:
        processed_data = preprocessor.fit_transform(df)
    else:
        processed_data = preprocessor.transform(df)
        
    return processed_data

if __name__ == '__main__':
    train_df = pd.read_csv('data/processed/train.csv')
    
    X_train = train_df.drop('credit_risk', axis=1)
    
    preprocessor_pipeline = build_feature_pipeline(X_train)
    preprocessor_pipeline.fit(X_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor_pipeline, 'models/preprocessor.joblib')
    
    print("Pipeline de preprocesamiento entrenado y guardado en 'models/preprocessor.joblib'")