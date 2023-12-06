from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import joblib

def load_and_predict(input_data, model_path='random_forest_model.joblib'):
    # Load the trained pipeline
    pipeline = joblib.load(model_path)

    # Preprocess the input data using the loaded pipeline
    preprocessed_data = preprocess_input_data(input_data, pipeline)

    # Make predictions using the loaded pipeline
    predictions = pipeline.predict(preprocessed_data)

    return predictions

def preprocess_input_data(input_data, pipeline):
    # Convert the input data to a DataFrame if it's not already in that format
    if isinstance(input_data, str):  # Check if the input_data is a file path
        if input_data.endswith('.csv'):
            input_df = pd.read_csv(input_data)
        elif input_data.endswith('.xlsx'):
            input_df = pd.read_excel(input_data)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel files are allowed.")
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("Input data must be a file path (CSV or Excel) or a DataFrame.")

    # Drop the target column if present
    if 'target_column_name' in input_df.columns:
        input_df = input_df.drop('target_column_name', axis=1)

    # Use the loaded pipeline to preprocess the input data
    preprocessed_data = pipeline['preprocessor'].transform(input_df)

    return preprocessed_data
