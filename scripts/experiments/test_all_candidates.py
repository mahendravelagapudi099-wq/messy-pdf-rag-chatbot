import sys
sys.path.insert(0, '.')
from src.rag import RAGEngine
from src.config import get_settings

settings = get_settings()
engine = RAGEngine(settings)

# Test all 50 candidate questions
with open('data/candidate_questions.jsonl') as f:
    questions = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        import json
        obj = json.loads(line)
        questions.append(obj)

results = []
for q in questions:
    qid = q['id']
    question = q['question']
    category = q['category']
    try:
        retrieval = engine.search(question)
        answer = engine.ask(q['question'])
        has_answer = answer.text != "" and not answer.abstained
        num_citations = len(answer.citations) if answer.citations else 0
        first_cit = answer.citations[0] if answer.citations else None
        results.append({
            'id': qid,
            'category': category,
            'question': question[:60] + "..." if len(question) > 60 else question,
            'has_answer': has_answer,
            'answer_text': answer.text[:80] if answer.text else "N/A",
            'abstained': answer.abstained,
            'num_citations': num_citations,
            'first_cit_filename': first_cit.filename if first_cit else None,
            'first_cit_page': first_cit.page if first_cit else None,
            'first_cit_sim': first_cit.similarity if first_cit else None,
        })
    except Exception as e:
        results.append({
            'id': qid,
            'category': category,
            'question': question[:60] + "..." if len(question) > 60 else question,
            'has_answer': False,
            'answer_text': f"Error: {e}",
            'abstained': True,
            'num_citations': 0,
            'first_cit_filename': None,
            'first_cit_page': None,
            'first_cit_sim': None,
        })

# Summary
total = len(results)
answerable = sum(1 for r in results if r['has_answer'] and not r['abstained'])
abstained = sum(1 for r in results if r['abstained'])
errors = sum(1 for r in results if not r['has_answer'] and not r['abstained'])

print(f"Total: {total}")
print(f"Answerable (extractive): {answerable}")
print(f"Abstained: {abstained}")
print(f"Errors: {errors}")
print()
print("=== DETAILED RESULTS ===")
for r in results:
    status = "ABSTAIN" if r['abstained'] else ("ANS" if r['has_answer'] else "ERROR")
    print(f"{r['id']:15} {status:8} {r['category']:12} sim={r['first_cit_sim']:.2f}" if r['first_cit_sim'] else 
          f"{r['id']:15} {status:8} {r['category']:12}" if r['has_answer'] else
          f"{r['id']:15} {status:8} {r['category']:12}")
    if r['first_cit_filename']:
        print(f"  -> {r['first_cit_filename']} p.{r['first_cit_page']} (sim={r['first_cit_sim']:.2f})")
    if r['answer_text'] and r['answer_text'] != "N/A" and not r['abstained']:
        print(f"  Answer: {r['answer_text'][:100]}")