from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Memuat model yang sudah dilatih di Colab
with open('model (6).pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "API Kredit Risiko Aktif!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Mengambil 4 fitur utama dari request React
        annual_income = float(data['annual_income'])
        total_outstanding_debt = float(data['total_outstanding_debt'])
        loan_amount = float(data['loan_application_amount'])
        credit_score = float(data['credit_score'])
        
        # Membuat DataFrame dengan susunan 4 kolom yang persis sama dengan saat training
        input_data = pd.DataFrame(
            [[annual_income, total_outstanding_debt, loan_amount, credit_score]], 
            columns=['annual_income', 'total_outstanding_debt', 'loan_application_amount', 'credit_score']
        )
        
        # Melakukan prediksi
        prediction = model.predict(input_data)
        
        hasil_teks = "Risiko Tinggi (Ditolak)" if prediction[0] == 1 else "Risiko Rendah (Disetujui)"
        
        return jsonify({'prediksi': hasil_teks})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
