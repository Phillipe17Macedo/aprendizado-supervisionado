from flask import Flask, render_template, request
from pypmml import Model
import pandas as pd
import os

app = Flask(__name__)

# Carregar o modelo PMML uma vez, na inicialização do aplicativo
model_path = os.path.join(os.path.dirname(__file__), 'pmml_arqv_rendom_forest_learner.pmml')
model = Model.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Dados de entrada ajustados para o modelo PMML
        input_data = {
            'region': request.form.get('region'),
            'powertrain': request.form.get('powertrain'),
            'year': float(request.form.get('year')),
            'value': float(request.form.get('value'))
        }

        # DataFrame para o modelo prever o 'mode'
        data = pd.DataFrame([input_data])

        # Debug: Exibir os dados de entrada
        print("Dados de entrada para predição:", input_data)

        # Fazer a previsão com o modelo carregado
        result = model.predict(data)

        # Debug: Exibir todos os resultados retornados pelo modelo
        print("Resultado completo do modelo:", result)

        # Critério ajustado para "Carro" ou "Ônibus"
        predicted_mode = 'Carro' if result['value*'].iloc[0] > -0.0717 else 'Ônibus'

        # Passar os valores para o template HTML
        output = {
            'region': input_data['region'],
            'year': input_data['year'],
            'value': input_data['value'],
            'prediction': predicted_mode  # "Carro" ou "Ônibus"
        }

        return render_template('result.html', result=output)
    
    except Exception as e:
        print(f"Erro ao fazer a previsão: {e}")
        return "Erro na previsão."

if __name__ == '__main__':
    app.run(debug=True)
