import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_process_data(raw_data_path, processed_folder):
    """Carga los datos crudos, les asigna nombres de columna y los guarda procesados."""
    # Nombres de las columnas según la documentación de german.doc
    column_names = [
        'existing_checking_account', 'duration_in_month', 'credit_history', 'purpose',
        'credit_amount', 'savings_account_bonds', 'present_employment_since',
        'installment_rate_in_percentage_of_disposable_income', 'personal_status_and_sex',
        'other_debtors_guarantors', 'present_residence_since', 'property', 'age_in_years',
        'other_installment_plans', 'housing', 'number_of_existing_credits_at_this_bank',
        'job', 'number_of_people_being_liable_to_provide_maintenance_for', 'telephone',
        'foreign_worker', 'credit_risk'
    ]
    
    df = pd.read_csv(raw_data_path, header=None, names=column_names, sep=' ')
    # Mapear la columna 'credit_risk' de 1 y 2 a 0 y 1
    df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})
    
    os.makedirs(processed_folder, exist_ok=True)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['credit_risk'])
    
    train_df.to_csv(os.path.join(processed_folder, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(processed_folder, 'test.csv'), index=False)
    
    print("Datos procesados y guardados en 'data/processed/'")

if __name__ == '__main__':
    load_and_process_data('data/raw/german.data.csv', 'data/processed')