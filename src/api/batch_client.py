import requests
import pandas as pd
import time
from tqdm import tqdm

API_URL = "http://127.0.0.1:8000/predict"

def run_batch_inference():
    """
    Lee los datos de prueba, los envía a la API para predicción en lote,
    y muestra los resultados.
    """
    print("Iniciando batch inference...")

    # Cargar los datos de prueba
    try:
        test_data = pd.read_csv("data/processed/test.csv")
    except FileNotFoundError:
        print("Error: No se encontró 'data/processed/test.csv'.")
        print("Asegúrate de haber ejecutado 'python src/data/make_dataset.py' primero.")
        return

    if 'credit_risk' in test_data.columns:
        actual_labels = test_data['credit_risk']
        features_to_predict = test_data.drop('credit_risk', axis=1)
    else:
        actual_labels = None
        features_to_predict = test_data

    records = features_to_predict.to_dict(orient='records')
    print(f"Se enviarán {len(records)} registros para predicción.")

    predictions = []
    probabilities = []
    
    start_time = time.time()
    
    # Iterar sobre cada registro y llamar a la API
    for record in tqdm(records, desc="Procesando Lote"):
        try:
            response = requests.post(API_URL, json=record)
            
            if response.status_code == 200:
                result = response.json()
                predictions.append(result['prediction'][0])
                probabilities.append(result['probability'][0])
            else:
                print(f"Error en el registro: {record}. Status: {response.status_code}, Body: {response.text}")
                predictions.append(None)
                probabilities.append(None)

        except requests.exceptions.RequestException as e:
            print(f"Error de conexión: {e}")
            break # Detener el proceso si la API no está disponible

    end_time = time.time()
    
    # 3. Procesar y mostrar los resultados
    duration = end_time - start_time
    records_processed = len(predictions)
    
    print("\n Inferencia por lotes completada.")
    print(f" Tiempo total: {duration:.2f} segundos.")
    if records_processed > 0:
        print(f" Velocidad: {records_processed / duration:.2f} predicciones/segundo.")
    
    results_df = features_to_predict.copy()
    results_df['predicted_risk'] = predictions
    results_df['predicted_probability_of_risk'] = probabilities
    
    if actual_labels is not None:
        results_df['actual_risk'] = actual_labels

    print("\n Muestra de los resultados de la predicción:")
    print(results_df.head())
    
    # results_df.to_csv("data/batch_predictions_output.csv", index=False)
    # print("\n💾 Resultados completos guardados en 'data/batch_predictions_output.csv'")


if __name__ == "__main__":
    run_batch_inference()