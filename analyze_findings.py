import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Read the CSV file
df = pd.read_csv('ai_bias_mitigation_research.csv')

# Analyze data by source
source_counts = df['source'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=source_counts.index, y=source_counts.values)
plt.title('Distribution of Data Sources')
plt.xlabel('Source')
plt.ylabel('Count')
plt.savefig('data_sources_distribution.png')
plt.close()

# Analyze data by company
company_counts = df['company'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=company_counts.index, y=company_counts.values)
plt.title('Distribution of Data by Company')
plt.xlabel('Company')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('company_data_distribution.png')
plt.close()

# Analyze most common words in titles
def get_word_counts(text_series):
    words = ' '.join(text_series).lower()
    words = re.findall(r'\w+', words)
    return Counter(words)

title_word_counts = get_word_counts(df['title'])
common_words = pd.DataFrame(title_word_counts.most_common(20), columns=['word', 'count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=common_words, x='word', y='count')
plt.title('Most Common Words in Titles')
plt.xlabel('Word')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('common_words_titles.png')
plt.close()

print('Analysis complete. Visualizations saved as PNG files.')
print('\nTop 20 most common words in titles:')
print(common_words)

# Sample of the data
print('\nSample of the collected data:')
print(df.sample(5).to_string(index=False))

# Analyze advancements in bias mitigation
def extract_advancements(text):
    keywords = ['new', 'novel', 'innovative', 'breakthrough', 'advanced', 'improvement']
    return isinstance(text, str) and any(keyword in text.lower() for keyword in keywords)

advancements = df[df['snippet'].apply(extract_advancements)]
print(f'\nNumber of entries mentioning advancements: {len(advancements)}')
print('\nSample of advancements in bias mitigation:')
print(advancements[['company', 'title']].sample(5).to_string(index=False))

# Save analysis results
with open('analysis_results.txt', 'w') as f:
    f.write(f'Total records: {len(df)}\n')
    f.write(f'Records by source:\n{source_counts}\n\n')
    f.write(f'Records by company:\n{company_counts}\n\n')
    f.write(f'Top 20 most common words in titles:\n{common_words}\n\n')
    f.write(f'Number of entries mentioning advancements: {len(advancements)}\n')

print('\nAnalysis results saved to analysis_results.txt')
