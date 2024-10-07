import pandas as pd
import numpy as np

# Generate synthetic ethical dilemmas and expert assessments
n_dilemmas = 1000
dilemmas = [f'Ethical dilemma {i}' for i in range(n_dilemmas)]
assessments = np.random.rand(n_dilemmas, 5)  # 5 different ethical dimensions

df = pd.DataFrame(assessments, columns=['Fairness', 'Transparency', 'Privacy', 'Accountability', 'Beneficence'])
df['Dilemma'] = dilemmas

df.to_csv('datasets/ai_ethics_dataset.csv', index=False)

print('AI ethics dataset created and saved.')
