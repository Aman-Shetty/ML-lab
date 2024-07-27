import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.read_csv('heart.csv')[['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'target']]
print(data.head())

model = BayesianNetwork([('age', 'target'), ('sex', 'target'), ('cp', 'target'),
                         ('thalach', 'target'), ('exang', 'target'), ('oldpeak', 'target')])
model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)
evidence = {'age': 63, 'sex': 1, 'cp': 1, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3}
result = inference.query(variables=['target'], evidence=evidence)
print(result)
