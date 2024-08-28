from flask import Flask, render_template, request
from pypmml import Model
import pandas as pd
import os

app = Flask(__name__)

# Carregar o modelo PMML uma vez, na inicialização do aplicativo
model_path = os.path.join(os.path.dirname(__file__), 'pmml_arqv_rendom_forest_learner.pmml')
model = Model.load(model_path)

def desnormalize_value(norm_value, orig_min, orig_max, norm_min, norm_max):
    """Desnormaliza o valor usando a fórmula inversa da LinearNorm."""
    return orig_min + ((norm_value - norm_min) * (orig_max - orig_min)) / (norm_max - norm_min)

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

        # Fazer a previsão com o modelo carregado
        result = model.predict(data)

        # Debug: Exibir todos os resultados retornados pelo modelo
        print("Resultado completo do modelo:", result)

        # Desnormalizar os resultados para torná-los interpretáveis
        desnormalized_year = desnormalize_value(
            result['year*'][0],
            orig_min=2010.0, orig_max=2035.0,
            norm_min=-383.1185786060764, norm_max=-382.9288625763933
        )
        desnormalized_value = desnormalize_value(
            result['value*'][0],
            orig_min=1.29999998534913E-5, orig_max=4.4E8,
            norm_min=-0.07227711653569541, norm_max=-0.07227700451014178
        )

        # Debug: Exibir os valores desnormalizados
        print("Year desnormalizado:", desnormalized_year)
        print("Value desnormalizado:", desnormalized_value)

        # Passar os valores desnormalizados para o template HTML
        output = {
            'region': input_data['region'],
            'year': input_data['year'],
            'value': input_data['value'],
            'prediction': {
                'Desnormalized Year': desnormalized_year,
                'Desnormalized Value': desnormalized_value
            }
        }

        return render_template('result.html', result=output)
    
    except Exception as e:
        print(f"Erro ao fazer a previsão: {e}")
        return "Erro na previsão."

if __name__ == '__main__':
    app.run(debug=True)