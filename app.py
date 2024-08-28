from pypmml import Model
import pandas as pd

# Carregar o modelo PMML
model = Model.load('pmml_arqv.pmml')

# Dados de exemplo
data = pd.DataFrame({
    'region': ['USA'],
    'category': ['Historical'],
    'parameter': ['EV sales'],
    'mode': ['Cars'],
    'mode_int': ['y'],
    'powertrain': ['EV'],
    'year': [2025],
    'unit': ['percent'],
    'value': [0.1]
})

# Preprocessar os dados conforme necessário (por exemplo, normalizar)
# O PMML deve ter essas etapas, mas você pode precisar aplicar manualmente

# Fazer previsões
predictions = model.predict(data)

# Exibir previsões
print(predictions)