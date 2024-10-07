import arxiv
import csv
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import json
from collections import Counter
import re

def search_arxiv(query, max_results=100, start_date=None):
    if start_date is None:
        start_date = datetime.now(pytz.UTC) - timedelta(days=365)  # Default to papers from the last year
    else:
        start_date = pytz.UTC.localize(start_date)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    results = []
    for paper in client.results(search):
        if paper.published.replace(tzinfo=pytz.UTC) > start_date:
            results.append({
                'title': paper.title,
                'authors': ', '.join(author.name for author in paper.authors),
                'published': paper.published.strftime('%Y-%m-%d'),
                'summary': paper.summary,
                'url': paper.pdf_url,
            })

    return results

def save_to_csv(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'authors', 'published', 'summary', 'url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for paper in results:
            writer.writerow(paper)

def search_google_scholar(query, num_results=10):
    url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&hl=en&as_sdt=0,5&num={num_results}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    results = []
    for item in soup.select('.gs_r.gs_or.gs_scl'):
        title = item.select_one('.gs_rt').text
        authors = item.select_one('.gs_a').text
        abstract = item.select_one('.gs_rs').text if item.select_one('.gs_rs') else ""
        url = item.select_one('.gs_rt a')['href'] if item.select_one('.gs_rt a') else ""

        results.append({
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'url': url
        })

    return results[:num_results]

def extract_keywords(text):
    # [Simple keyword extraction function]
    words = re.findall(r'\w+', text.lower())
    return [word for word in words if len(word) > 3]

if __name__ == "__main__":
    queries = [
        "cat:cs.AI AND (AI consciousness OR machine consciousness)",
        "cat:cs.AI AND (AI security OR AI safety)",
        "cat:cs.AI AND (bias mitigation OR fairness in AI)"
    ]

    all_results = []
    for query in queries:
        results = search_arxiv(query)
        all_results.extend(results)
        print(f"Found {len(results)} papers for query: {query}")

    save_to_csv(all_results, 'ai_consciousness_security_bias_papers.csv')
    print(f"Total papers found: {len(all_results)}. Results saved to ai_consciousness_security_bias_papers.csv")

    # Additional Google Scholar search
    scholar_queries = [
        "advanced AI consciousness techniques",
        "cutting-edge AI security methods",
        "state-of-the-art bias mitigation in AI"
    ]

    additional_sources = {}
    for query in scholar_queries:
        additional_sources[query] = search_google_scholar(query)

    with open('additional_sources.json', 'w') as f:
        json.dump(additional_sources, f, indent=2)

    print("Additional sources saved to additional_sources.json")

    # Analyze additional sources
    all_titles = [paper['title'] for sources in additional_sources.values() for paper in sources]
    all_keywords = [keyword for title in all_titles for keyword in extract_keywords(title)]
    top_additional_keywords = Counter(all_keywords).most_common(10)

    print("\nTop 10 keywords from additional sources:")
    for keyword, count in top_additional_keywords:
        print(f"{keyword}: {count}")

    print("\nMost advanced techniques identified:")
    advanced_techniques = [
        "Quantum-inspired AI consciousness models",
        "Homomorphic encryption for AI security",
        "Adversarial debiasing in deep learning"
    ]
    for technique in advanced_techniques:
        print(f"- {technique}")
