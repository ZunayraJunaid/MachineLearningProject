# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
from preprocessing import preprocess_input

app = Flask(__name__)

# Load best model
with open('models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    try:
        input_scaled = preprocess_input(form_data)
        prediction  = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        result = {
            'prediction': 'CKD Detected' if prediction == 1 else 'No CKD Detected',
            'confidence': round(max(probability) * 100, 2),
            'risk'      : 'High Risk' if prediction == 1 else 'Low Risk',
            'form_data' : form_data
        }
    except Exception as e:
        result = {'error': str(e)}

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)