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
            'region': request.form.get('region', 'Europe'),
            'category': request.form.get('category', 'Historical'),
            'parameter': request.form.get('parameter', 'EV stock share'),
            'mode': request.form.get('mode', 'Cars'),
            'powertrain': request.form.get('powertrain', 'EV'),
            'year': float(request.form.get('year')),
            'unit': request.form.get('unit', 'percent'),
            'value': float(request.form.get('value'))
        }

        # Debug: Exibir os dados de entrada
        print("Dados de entrada para predição:", input_data)

        # Preparar os dados para o modelo
        data = pd.DataFrame([input_data])

        # Debug: Exibir o DataFrame preparado para o modelo
        print("DataFrame para predição:", data)

        # Fazer a previsão com o modelo carregado
        result = model.predict(data)

        # Debug: Exibir todos os resultados retornados pelo modelo
        print("Resultado completo do modelo:", result)

        # Obter a predição (ajuste conforme necessário)
        prediction = result.iloc[0].to_dict()

        # Passar os valores para o template HTML
        output = {
            'region': input_data['region'],
            'year': input_data['year'],
            'value': input_data['value'],
            'prediction': prediction  # Ajuste para a classe prevista
        }

        return render_template('result.html', result=output)
    
    except Exception as e:
        print(f"Erro ao fazer a previsão: {e}")
        return "Erro na previsão."

if __name__ == '__main__':
    app.run(debug=True)