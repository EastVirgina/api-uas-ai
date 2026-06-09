from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app) # Penting agar React bisa menembak API ini tanpa error CORS

# Load model yang sudah di-download
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "API Machine Learning Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Menangkap data JSON dari React
        data = request.get_json()
        
        # Ekstrak 4 parameter wajib sesuai tugas
        fitur1 = float(data['fitur1'])
        fitur2 = float(data['fitur2'])
        fitur3 = float(data['fitur3'])
        fitur4 = float(data['fitur4'])
        
        # Ubah ke format DataFrame agar sesuai dengan input model
        input_data = pd.DataFrame([[fitur1, fitur2, fitur3, fitur4]], 
                                  columns=['customer_id', 'gender', 'employment_status', 'annual_income', 'avg_monthly_balance', 'total_outstanding_debt', 'loan_application_amount', 'age', 'late_payment_count', 'credit_score', 'credit_risk'])
        
        # Lakukan prediksi
        hasil = model.predict(input_data)
        
        # Kembalikan hasil ke React
        return jsonify({'prediksi': str(hasil[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)