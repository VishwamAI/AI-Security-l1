import pandas as pd
import numpy as np

# Generate synthetic security threats and attacks data
n_samples = 5000
threat_types = ['DDoS', 'SQL Injection', 'XSS', 'Phishing', 'Malware']
data = {
    'threat_type': np.random.choice(threat_types, n_samples),
    'severity': np.random.randint(1, 11, n_samples),
    'success_rate': np.random.rand(n_samples)
}

df = pd.DataFrame(data)
df.to_csv('datasets/ai_security_benchmark.csv', index=False)

print('AI security benchmark dataset created and saved.')
