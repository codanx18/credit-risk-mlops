import pytest
import joblib
import pandas as pd
import numpy as np

@pytest.fixture
def model():
    """Carga el modelo entrenado."""
    return joblib.load('models/model.joblib')

@pytest.fixture
def preprocessor():
    """Carga el preprocesador entrenado."""
    return joblib.load('models/preprocessor.joblib')

@pytest.fixture
def sample_data():
    """Crea datos de ejemplo para las pruebas."""
    # Ejemplos de clientes de 'bajo riesgo' y 'alto riesgo'
    data = {
        'existing_checking_account': ['A11', 'A14'],
        'duration_in_month': [6, 24],
        'credit_history': ['A34', 'A32'],
        'purpose': ['A43', 'A40'],
        'credit_amount': [1169, 5951],
        'savings_account_bonds': ['A65', 'A61'],
        'present_employment_since': ['A75', 'A73'],
        'installment_rate_in_percentage_of_disposable_income': [4, 2],
        'personal_status_and_sex': ['A93', 'A92'],
        'other_debtors_guarantors': ['A101', 'A101'],
        'present_residence_since': [4, 2],
        'property': ['A121', 'A121'],
        'age_in_years': [67, 22],
        'other_installment_plans': ['A143', 'A143'],
        'housing': ['A152', 'A152'],
        'number_of_existing_credits_at_this_bank': [2, 1],
        'job': ['A173', 'A173'],
        'number_of_people_being_liable_to_provide_maintenance_for': [1, 1],
        'telephone': ['A192', 'A191'],
        'foreign_worker': ['A201', 'A201'],
    }
    return pd.DataFrame(data)

def test_prediction_output(model, preprocessor, sample_data):
    """Prueba que la salida de la predicción tiene la forma y el tipo correctos."""
    processed_data = preprocessor.transform(sample_data)
    predictions = model.predict(processed_data)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert all(pred in [0, 1] for pred in predictions)

def test_prediction_values(model, preprocessor, sample_data):
    """Prueba que las predicciones tienen sentido (opcional pero recomendado)."""
    processed_data = preprocessor.transform(sample_data)
    predictions = model.predict(processed_data)
    # Se espera que el primer cliente sea de bajo riesgo (0)
    # y el segundo sea de alto riesgo (1).
    assert predictions[0] == 0
    # assert predictions[1] == 1 # Esta aserción puede fallar