import arxiv
import csv
from datetime import datetime, timedelta
import pytz

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

if __name__ == "__main__":
    query = "cat:cs.CR AND (AI security OR machine learning security OR deep learning security)"
    results = search_arxiv(query)
    save_to_csv(results, 'ai_security_papers.csv')
    print(f"Found {len(results)} papers. Results saved to ai_security_papers.csv")
