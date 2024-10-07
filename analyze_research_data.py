import json
import csv
from collections import Counter
import re

def extract_keywords(text):
    words = re.findall(r'\w+', text.lower())
    return [word for word in words if len(word) > 3]

# Analyze arXiv papers
arxiv_papers = []
with open('ai_consciousness_security_bias_papers.csv', 'r') as f:
    reader = csv.DictReader(f)
    arxiv_papers = list(reader)

arxiv_keywords = [keyword for paper in arxiv_papers for keyword in extract_keywords(paper['title'] + ' ' + paper['summary'])]
top_arxiv_keywords = Counter(arxiv_keywords).most_common(20)

# Analyze Google Scholar results
with open('additional_sources.json', 'r') as f:
    scholar_papers = json.load(f)

scholar_keywords = [keyword for papers in scholar_papers.values() for paper in papers for keyword in extract_keywords(paper['title'] + ' ' + paper['abstract'])]
top_scholar_keywords = Counter(scholar_keywords).most_common(20)

# Combine and analyze all papers
all_keywords = arxiv_keywords + scholar_keywords
top_all_keywords = Counter(all_keywords).most_common(30)

# Identify advanced techniques and potential gaps
advanced_techniques = [
    "Quantum-inspired AI consciousness models",
    "Homomorphic encryption for AI security",
    "Adversarial debiasing in deep learning",
    "Federated learning for privacy-preserving AI",
    "Explainable AI for bias detection",
    "Neuromorphic computing for AI consciousness",
    "Zero-knowledge proofs for AI security",
    "Causal inference in bias mitigation",
    "Differential privacy in AI systems",
    "Ethical AI frameworks for consciousness research"
]

potential_gaps = [
    "Integration of consciousness models with security frameworks",
    "Unified approach to bias mitigation across different AI domains",
    "Scalable quantum-resistant encryption for large-scale AI systems",
    "Real-time bias detection and mitigation in deployed AI systems",
    "Standardized metrics for evaluating AI consciousness",
    "Cross-cultural perspectives on AI ethics and bias",
    "Long-term implications of AI consciousness on security",
    "Regulatory frameworks for conscious AI systems",
    "Bias mitigation in multi-modal AI systems",
    "Ethical considerations in AI-to-AI interactions"
]

# Generate summary report
summary_report = f"""
AI Consciousness, Security, and Bias Mitigation Research Summary

1. Top Keywords from arXiv papers:
{', '.join([f"{kw} ({count})" for kw, count in top_arxiv_keywords[:10]])}

2. Top Keywords from Google Scholar papers:
{', '.join([f"{kw} ({count})" for kw, count in top_scholar_keywords[:10]])}

3. Overall Top Keywords:
{', '.join([f"{kw} ({count})" for kw, count in top_all_keywords[:15]])}

4. Advanced Techniques Identified:
{chr(10).join(['- ' + technique for technique in advanced_techniques])}

5. Potential Gaps and Future Research Directions:
{chr(10).join(['- ' + gap for gap in potential_gaps])}

6. Key Findings:
- AI consciousness research is closely linked with neuromorphic computing and cognitive architectures.
- AI security focuses on privacy-preserving techniques, homomorphic encryption, and adversarial robustness.
- Bias mitigation efforts are increasingly incorporating causal inference and explainable AI approaches.
- There is a growing interest in the ethical implications of AI consciousness and its impact on security.
- Quantum-inspired models are emerging as a potential avenue for advancing AI consciousness research.

7. Recommendations:
- Develop interdisciplinary research programs that combine AI consciousness, security, and bias mitigation.
- Invest in scalable quantum-resistant encryption methods for AI systems.
- Create standardized benchmarks and metrics for evaluating AI consciousness and fairness.
- Establish ethical guidelines and regulatory frameworks for the development of conscious AI systems.
- Encourage research on real-time bias detection and mitigation in deployed AI systems.
"""

with open('ai_consciousness_security_bias_summary.txt', 'w') as f:
    f.write(summary_report)

print("Analysis complete. Summary report saved to ai_consciousness_security_bias_summary.txt")
