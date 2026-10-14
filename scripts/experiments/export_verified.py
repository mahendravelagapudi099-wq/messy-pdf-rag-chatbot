import json
import sys
sys.path.insert(0, '.')

# Read the candidate questions
with open('data/candidate_questions.jsonl') as f:
    questions = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # Reformatting to match the eval dataset schema
        questions.append({
            "id": obj["id"],
            "category": obj["category"],
            "question": obj["question"],
            "reference_answer": obj.get("reference_answer", ""),
            "expected_source": obj.get("expected_source"),
            "expected_pages": obj.get("expected_pages", []),
            "required_facts": obj.get("required_facts", []),
            "should_abstain": obj.get("should_abstain", False),
        })

# Write in the eval dataset format
with open('data/real_verified_questions.jsonl', 'w') as f:
    for q in questions:
        f.write(json.dumps(q) + '\n')

print(f'Wrote {len(questions)} questions to data/real_verified_questions.jsonl')