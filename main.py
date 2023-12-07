from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

app = Flask(__name__)

# Load the trained pipeline when the app starts
pipeline = joblib.load('random_forest_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', message='No selected file')

    try:
        # input file is a CSV or Excel file
        input_data = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
    except pd.errors.EmptyDataError:
        return render_template('index.html', message='File is empty or invalid format')
    except pd.errors.ParserError:
        return render_template('index.html', message='Error parsing file')

    # Preprocess the input data using the loaded pipeline
    preprocessed_data = pipeline.named_steps['preprocessor'].transform(input_data)

    # Make predictions using the loaded pipeline
    predictions = pipeline.named_steps['classifier'].predict(preprocessed_data)

    # Add the predictions to the DataFrame
    input_data['predictions'] = predictions

    # Save the results to a new CSV file
    result_file_path = 'predicted_results.csv'
    input_data.to_csv(result_file_path, index=False)

    return send_file(result_file_path, as_attachment=True, download_name='predicted_results.csv')

if __name__ == '__main__':
    app.run(debug=True)
