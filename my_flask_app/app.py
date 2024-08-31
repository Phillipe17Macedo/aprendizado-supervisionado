from flask import Flask, render_template, request
from pypmml import Model
import pandas as pd
import os

app = Flask(__name__)

# Carregar o modelo PMML
model_path = os.path.join(os.path.dirname(__file__), 'pmml_arqv_decision_tree_learner.pmml')
model = Model.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capturar os dados de entrada do formulário
        input_data = {
            'Ratings': float(request.form.get('ratings')) / 5,  # Normaliza para 0-1
            'RAM': float(request.form.get('ram')) / 16,         # Normaliza para 0-1
            'ROM': float(request.form.get('rom')) / 512,        # Normaliza para 0-1
            'Primary_Cam': float(request.form.get('primary_cam')) / 108,  # Normaliza para 0-1
            'Selfi_Cam': float(request.form.get('selfi_cam')) / 32,       # Normaliza para 0-1
            'Battery_Power': float(request.form.get('battery_power')) / 6000  # Normaliza para 0-1
        }

        # Criar um DataFrame para a predição
        data = pd.DataFrame([input_data])

        # Debug: Mostrar os dados de entrada para a predição
        print("Dados de entrada para predição:", input_data)

        # Fazer a predição com o modelo carregado
        result = model.predict(data)

        # Debug: Mostrar os resultados completos retornados pelo modelo
        print("Resultado completo do modelo:", result)

        # Obter a predição correta do campo 'predicted_category'
        predicted_category = result['predicted_category'][0]

        # Passar os valores para o template HTML
        output = {
            'ratings': request.form.get('ratings'),
            'ram': request.form.get('ram'),
            'rom': request.form.get('rom'),
            'primary_cam': request.form.get('primary_cam'),
            'selfi_cam': request.form.get('selfi_cam'),
            'battery_power': request.form.get('battery_power'),
            'prediction': predicted_category
        }

        return render_template('result.html', result=output)

    except Exception as e:
        print(f"Erro ao fazer a previsão: {e}")
        return "Erro na previsão."

if __name__ == '__main__':
    app.run(debug=True)