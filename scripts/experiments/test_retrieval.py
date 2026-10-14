import sys
sys.path.insert(0, '.')
from src.rag import RAGEngine
from src.config import get_settings

settings = get_settings()
engine = RAGEngine(settings)

# Test a few questions against the real corpus
test_questions = [
    "What is a user allowed to do without having to establish existence of undocumented features?",
    "What positions can a user take under the RBI Master Direction?",
    "What is the circular number and date of this RBI due diligence circular?",
]

for q in test_questions:
    print(f'Q: {q}')
    try:
        retrieval = engine.search(q)
        answer = engine.ask(q)
        print(f'A: {answer.text[:200]}')
        print(f'Citations: {len(answer.citations)}')
        for c in answer.citations[:3]:
            print(f'  - {c.filename} p.{c.page} sim={c.similarity:.3f}')
    except Exception as e:
        print(f'Error: {e}')
    print()