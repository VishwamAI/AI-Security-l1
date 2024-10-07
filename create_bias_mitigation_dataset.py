import pandas as pd
import numpy as np

# Generate synthetic data for text, image, and demographic information
n_samples = 5000
text_data = np.random.rand(n_samples, 100)
image_data = np.random.rand(n_samples, 50, 50, 3)
demographics = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['M', 'F', 'O'], n_samples),
    'ethnicity': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
})

# Save datasets
np.save('datasets/text_data.npy', text_data)
np.save('datasets/image_data.npy', image_data)
demographics.to_csv('datasets/demographics.csv', index=False)

print('Diverse bias mitigation dataset created and saved.')
