import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv('dataset.csv')
heartDisease = heartDisease.replace('?', np.nan)

for column in heartDisease.columns:
    heartDisease[column] = pd.to_numeric(heartDisease[column], errors='coerce')

heartDisease.dropna(inplace=True)

model = DiscreteBayesianNetwork([
    ('age', 'fbs'),
    ('fbs', 'target'),
    ('target', 'restecg'),
    ('target', 'thalach'),
    ('target', 'chol')
])

model.fit(data=heartDisease)

infer = VariableElimination(model)
q = infer.query(variables=['target'], evidence={'age': 37})
print(q)
