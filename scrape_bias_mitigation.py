import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_company_info(company_name):
    queries = [
        f'{company_name} AI bias mitigation',
        f'{company_name} fairness in machine learning',
        f'{company_name} ethical AI development'
    ]
    results = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for query in queries:
        url = f'https://www.google.com/search?q={quote_plus(query)}'
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            for g in soup.find_all('div', class_='g'):
                anchors = g.find_all('a')
                if anchors:
                    link = anchors[0]['href']
                    title = g.find('h3')
                    title = title.text if title else 'N/A'
                    snippet = g.find('div', class_='VwiC3b')
                    snippet = snippet.text if snippet else 'N/A'
                    results.append({'title': title, 'link': link, 'snippet': snippet, 'query': query})
        except requests.RequestException as e:
            logging.error(f"Error scraping {company_name} with query '{query}': {str(e)}")

    return results

def scrape_arxiv(query):
    url = f'http://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results=10'
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'xml')
        results = []
        for entry in soup.find_all('entry'):
            title = entry.title.text if entry.title else 'N/A'
            link = entry.id.text if entry.id else 'N/A'
            summary = entry.summary.text if entry.summary else 'N/A'
            results.append({'title': title, 'link': link, 'snippet': summary, 'source': 'arXiv'})
        return results
    except requests.RequestException as e:
        logging.error(f"Error scraping arXiv with query '{query}': {str(e)}")
        return []

def scrape_news(query):
    url = f'https://news.google.com/rss/search?q={quote_plus(query)}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'xml')
        results = []
        for item in soup.find_all('item'):
            title = item.title.text if item.title else 'N/A'
            link = item.link.text if item.link else 'N/A'
            description = item.description.text if item.description else 'N/A'
            results.append({'title': title, 'link': link, 'snippet': description, 'source': 'News'})
        return results
    except requests.RequestException as e:
        logging.error(f"Error scraping news with query '{query}': {str(e)}")
        return []

companies = ['IBM', 'Google', 'Meta', 'OpenAI', 'Apple', 'Microsoft', 'Amazon', 'DeepMind', 'NVIDIA', 'Anthropic']
all_results = []

for company in companies:
    logging.info(f'Scraping data for {company}...')
    company_results = scrape_company_info(company)
    for result in company_results:
        result['company'] = company
        result['source'] = 'Company Search'
    all_results.extend(company_results)
    time.sleep(2)  # Add a delay to avoid overwhelming the server

# Scrape arXiv for general AI bias mitigation papers
arxiv_query = "AI bias mitigation OR fairness in machine learning"
arxiv_results = scrape_arxiv(arxiv_query)
all_results.extend(arxiv_results)

# Scrape news articles
news_query = "AI bias mitigation ethical AI development"
news_results = scrape_news(news_query)
all_results.extend(news_results)

df = pd.DataFrame(all_results)
df.to_csv('ai_bias_mitigation_research.csv', index=False)
logging.info('Data collection complete. Results saved to ai_bias_mitigation_research.csv')

# Print summary of collected data
print(f"Total records collected: {len(df)}")
print(f"Records by source:")
print(df['source'].value_counts())
print(f"\nRecords by company:")
print(df['company'].value_counts())
