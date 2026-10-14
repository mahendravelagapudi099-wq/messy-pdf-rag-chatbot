import sys
sys.path.insert(0, '.')
from src.evaluation.benchmark import run_benchmark, BenchmarkConfig
from src.evaluation.dataset import load_dataset, coverage
from src.evaluation.metrics import Metrics

# Load the verified questions
questions = load_dataset('data/real_verified_questions.jsonl')
print(f'Loaded {len(questions)} questions')

# Just run extractive mode with one config (256/32)
results = run_benchmark(
    questions=questions,
    corpus_dir='data/real_corpus',
    mode='extractive',
    reranker=False,
    chunk_size=256,
    chunk_overlap=32,
)

metrics = results.metrics
print(f'\n=== Results for chunk_size=256, chunk_overlap=32 ===')
print(f'Hit Rate@5: {metrics.hit_rate_at_5:.4f}')
print(f'MRR: {metrics.mrr:.4f}')
print(f'Answer F1: {metrics.answer_f1:.4f}')
print(f'Fact recall: {metrics.fact_recall:.4f}')
print(f'Citation correct: {metrics.citation_correct:.4f}')
print(f'Unsupported answer rate: {metrics.unsupported_answer_rate:.4f}')
print(f'Abstention rate: {metrics.abstention_rate:.4f}')

# Coverage summary
cov = coverage(questions)
print(f'\nCoverage summary:')
print(cov.summary())