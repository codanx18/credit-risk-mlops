import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Modelo de riesgo crediticio")

def train_model():
    """Entrena, evalúa y registra el modelo con MLflow."""
    
    train_df = pd.read_csv('data/processed/train.csv')
    X_train_raw = train_df.drop('credit_risk', axis=1)
    y_train = train_df['credit_risk']

    preprocessor = joblib.load('models/preprocessor.joblib')
    X_train = preprocessor.transform(X_train_raw)

    with mlflow.start_run():
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Mejores parámetros: {best_params}")
        mlflow.log_params(best_params)
        
        y_pred = best_model.predict(X_train)
        y_pred_proba = best_model.predict_proba(X_train)[:, 1]
        
        accuracy = accuracy_score(y_train, y_pred)
        roc_auc = roc_auc_score(y_train, y_pred_proba)
        f1 = f1_score(y_train, y_pred)
        
        print(f"Accuracy (Train): {accuracy:.4f}")
        print(f"ROC AUC (Train): {roc_auc:.4f}")
        print(f"F1-Score (Train): {f1:.4f}")
        
        mlflow.log_metric("train_accuracy", accuracy)
        mlflow.log_metric("train_roc_auc", roc_auc)
        mlflow.log_metric("train_f1_score", f1)
        
        mlflow.sklearn.log_model(best_model, "credit_risk_model")
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/model.joblib')
        
        print("Modelo entrenado, evaluado y registrado con MLflow.")

if __name__ == '__main__':
    train_model()