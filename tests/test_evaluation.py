"""Tests for evaluation metrics, dataset handling and benchmark wiring."""

from __future__ import annotations

import json

import pytest

from src.evaluation.dataset import (
    VALID_CATEGORIES,
    DatasetError,
    EvalQuestion,
    coverage,
    dump_dataset,
    load_dataset,
)
from src.evaluation.metrics import (
    Metrics,
    QuestionOutcome,
    aggregate,
    citation_correct,
    fact_recall,
    normalize_text,
    reciprocal_rank,
    retrieved_hit,
    token_f1,
)
from src.models import Answer, Chunk, Citation, RetrievedChunk


def rc(filename: str, page: int, text: str, sim: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(f"c{page}", "d", filename, page, text), similarity=sim)


class TestTokenF1:
    def test_identical_is_one(self):
        assert token_f1("the repo rate is 6.5", "the repo rate is 6.5") == 1.0

    def test_no_overlap_is_zero(self):
        assert token_f1("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_partial_overlap_is_between(self):
        f1 = token_f1("repo rate is 6.5 percent", "repo rate unchanged at 6.5 percent")
        assert 0.0 < f1 < 1.0

    def test_normalisation_removes_stopwords_and_case(self):
        assert token_f1("The Repo Rate", "repo rate") == 1.0

    def test_both_empty_is_one(self):
        assert token_f1("", "") == 1.0

    def test_normalize_text_drops_stopwords(self):
        assert "the" not in normalize_text("The quick fox")
        assert "quick" in normalize_text("The quick fox")


class TestFactRecall:
    def test_all_facts_present(self):
        assert fact_recall("The rate is 80 per cent", ["80 per cent"]) == 1.0

    def test_partial_facts(self):
        assert fact_recall("The rate is 80 per cent", ["80 per cent", "three years"]) == 0.5

    def test_no_facts_present(self):
        assert fact_recall("Something else", ["80 per cent"]) == 0.0

    def test_no_facts_required_is_undefined_not_zero(self):
        """Returning 0.0 would falsely punish questions without fact labels."""
        import math

        assert math.isnan(fact_recall("anything", []))

    def test_case_insensitive(self):
        assert fact_recall("REPO RATE", ["repo rate"]) == 1.0


class TestRetrievedHit:
    def test_hit_on_gold_document_and_page(self):
        assert retrieved_hit([rc("a.pdf", 3, "text")], "a.pdf", [3])

    def test_hit_within_page_tolerance(self):
        assert retrieved_hit([rc("a.pdf", 4, "text")], "a.pdf", [3], page_tolerance=1)

    def test_miss_on_wrong_page(self):
        assert not retrieved_hit([rc("a.pdf", 9, "text")], "a.pdf", [3])

    def test_miss_on_wrong_document(self):
        assert not retrieved_hit([rc("b.pdf", 3, "text")], "a.pdf", [3])

    def test_miss_on_empty_retrieval(self):
        assert not retrieved_hit([], "a.pdf", [1])

    def test_textual_overlap_route_without_page_gold(self):
        ref = "The vehicle retention rate must be 80 per cent after three years"
        assert retrieved_hit([rc("a.pdf", 1, ref)], None, [], reference=ref)

    def test_document_only_gold_ignores_pages(self):
        assert retrieved_hit([rc("a.pdf", 7, "text")], "a.pdf", [])


class TestReciprocalRank:
    def test_first_position_is_one(self):
        assert reciprocal_rank([rc("a.pdf", 1, "t")], "a.pdf", [1]) == 1.0

    def test_second_position_is_half(self):
        assert reciprocal_rank([rc("b.pdf", 1, "t"), rc("a.pdf", 1, "t")], "a.pdf", [1]) == 0.5

    def test_no_hit_is_zero(self):
        assert reciprocal_rank([rc("b.pdf", 1, "t")], "a.pdf", [1]) == 0.0

    def test_empty_is_zero(self):
        assert reciprocal_rank([], "a.pdf", [1]) == 0.0


class TestCitationCorrect:
    def _answer(self, filename, page) -> Answer:
        return Answer(
            question="q",
            text="answer [1]",
            citations=[
                Citation("[1]", "d", filename, page, "c1", "excerpt", 0.9)
            ],
        )

    def test_correct_citation(self):
        assert citation_correct(self._answer("a.pdf", 3), "a.pdf", [3]) is True

    def test_wrong_page(self):
        assert citation_correct(self._answer("a.pdf", 9), "a.pdf", [3]) is False

    def test_missing_citations_is_false(self):
        answer = Answer(question="q", text="no citations")
        assert citation_correct(answer, "a.pdf", [3]) is False

    def test_no_expectation_excluded(self):
        """Returns None so it is left out of the metric denominator."""
        assert citation_correct(self._answer("a.pdf", 1), None, []) is None


class TestAggregate:
    def _outcomes(self):
        return [
            QuestionOutcome("factual-1", "q1", False, hit=True, reciprocal_rank=1.0,
                            answer_f1=0.8, fact_recall=1.0, citation_correct=True),
            QuestionOutcome("factual-2", "q2", False, hit=False, reciprocal_rank=0.0,
                            answer_f1=0.2, fact_recall=0.2, citation_correct=False),
            QuestionOutcome("unansw-1", "q3", True, abstained=True),
            QuestionOutcome("unansw-2", "q4", True, abstained=False),
        ]

    def test_hit_rate(self):
        m = aggregate(self._outcomes())
        assert m.hit_rate_at_k == 0.5

    def test_mrr(self):
        assert aggregate(self._outcomes()).mrr == 0.5

    def test_abstention_rate_only_counts_unanswerable(self):
        assert aggregate(self._outcomes()).abstention_rate == 0.5

    def test_unsupported_rate(self):
        m = aggregate(self._outcomes(), support_threshold=0.5)
        # Both answerable questions produced an answer; one of them has
        # fact_recall 0.2 < 0.5, so exactly half the answers are unsupported.
        assert m.unsupported_rate == 0.5
        assert m.answerable_answered == 2

    def test_unsupported_rate_ignores_abstentions(self):
        """An abstention is not an unsupported answer; it is the correct behaviour."""
        outcomes = [
            QuestionOutcome("factual-1", "q1", False, fact_recall=0.0, abstained=True),
            QuestionOutcome("factual-2", "q2", False, fact_recall=1.0, abstained=False),
        ]
        m = aggregate(outcomes, support_threshold=0.5)
        assert m.unsupported_rate == 0.0
        assert m.answerable_answered == 1

    def test_counts_are_split_correctly(self):
        m = aggregate(self._outcomes())
        assert (m.n_questions, m.n_answerable, m.n_unanswerable) == (4, 2, 2)

    def test_ingestion_failure_rate(self):
        m = aggregate([], ingestion_total=10, ingestion_failed=2)
        assert m.ingestion_failure_rate == 0.2

    def test_empty_is_safe(self):
        m = aggregate([])
        assert m.n_questions == 0
        assert m.hit_rate_at_k == 0.0

    def test_as_dict_is_json_serialisable(self):
        json.dumps(aggregate(self._outcomes()).as_dict())


class TestDataset:
    def test_loads_committed_dataset(self, eval_dataset_path):
        questions = load_dataset(eval_dataset_path)
        assert len(questions) >= 50, "the dataset must support at least 50 questions"

    def test_all_categories_present(self, eval_dataset_path):
        cov = coverage(load_dataset(eval_dataset_path))
        assert cov.is_complete
        assert all(cov.by_category.get(c, 0) > 0 for c in VALID_CATEGORIES)

    def test_has_unanswerable_questions(self, eval_dataset_path):
        cov = coverage(load_dataset(eval_dataset_path))
        assert cov.unanswerable > 0

    def test_answerable_questions_have_ground_truth(self, eval_dataset_path):
        for q in load_dataset(eval_dataset_path):
            if q.is_answerable:
                assert q.expected_source, f"{q.id} has no expected_source"
                assert q.expected_pages, f"{q.id} has no expected_pages"
                assert q.required_facts, f"{q.id} has no required_facts"

    def test_unanswerable_have_no_reference(self, eval_dataset_path):
        for q in load_dataset(eval_dataset_path):
            if q.should_abstain:
                assert not q.reference_answer

    def test_expected_sources_exist_in_corpus(self, eval_dataset_path, demo_corpus):
        names = {p.name for p in demo_corpus.glob("*.pdf")}
        for q in load_dataset(eval_dataset_path):
            if q.expected_source:
                assert q.expected_source in names, f"{q.id} -> {q.expected_source}"

    def test_roundtrip(self, tmp_path, eval_dataset_path):
        original = load_dataset(eval_dataset_path)
        out = dump_dataset(original, tmp_path / "out.jsonl")
        assert len(load_dataset(out)) == len(original)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetError):
            load_dataset(tmp_path / "nope.jsonl")

    def test_malformed_line_skipped_in_lenient_mode(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(
            '{"id":"a","question":"q1"}\nNOT JSON\n{"id":"b","question":"q2"}\n',
            encoding="utf-8",
        )
        assert len(load_dataset(p)) == 2

    def test_malformed_line_raises_in_strict_mode(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"id":"a","question":"q1"}\nNOT JSON\n', encoding="utf-8")
        with pytest.raises(DatasetError):
            load_dataset(p, strict=True)

    def test_duplicate_ids_skipped(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(
            '{"id":"a","question":"q1"}\n{"id":"a","question":"q2"}\n', encoding="utf-8"
        )
        assert len(load_dataset(p)) == 1

    def test_empty_dataset_raises(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text("\n\n", encoding="utf-8")
        with pytest.raises(DatasetError):
            load_dataset(p)

    def test_expected_pages_accepts_scalar(self):
        q = EvalQuestion.from_dict({"id": "x", "question": "y", "expected_pages": 4})
        assert q.expected_pages == [4]


class TestBenchmarkWiring:
    def test_config_key_is_unique_per_chunking(self):
        from src.evaluation import BenchmarkConfig

        keys = {
            BenchmarkConfig(
                dataset_path="x", corpus_dir="y", chunk_size=s, chunk_overlap=o
            ).key()
            for s, o in ((256, 32), (512, 64), (1024, 128))
        }
        assert len(keys) == 3

    def test_reranker_changes_key(self):
        from src.evaluation import BenchmarkConfig

        base = BenchmarkConfig(dataset_path="x", corpus_dir="y")
        rr = BenchmarkConfig(dataset_path="x", corpus_dir="y", use_reranker=True)
        assert base.key() != rr.key()

    def test_three_experiment_configs_defined(self):
        from src.config import CHUNKING_EXPERIMENTS

        assert len(CHUNKING_EXPERIMENTS) >= 3
        for cfg in CHUNKING_EXPERIMENTS:
            assert cfg["chunk_overlap"] < cfg["chunk_size"]


class TestExtractiveAbstention:
    """The extractive answerer must signal abstention, not return a wrong answer."""

    def test_abstains_when_no_overlap(self):
        from src.generation import extractive_answer

        chunks = [
            rc("a.pdf", 1, "Thermodynamic system, surroundings and boundary conditions.")
        ]
        result = extractive_answer("What is the repo rate decided by the MPC?", chunks)
        assert result.abstained
        assert result.reason

    def test_abstains_with_no_evidence(self):
        from src.generation import extractive_answer

        assert extractive_answer("anything", []).abstained

    def test_answers_with_matching_evidence(self):
        from src.generation import extractive_answer

        chunks = [
            rc("a.pdf", 1, "The vehicle retention rate shall be not less than 80 per cent "
                           "after three years of operation.")
        ]
        result = extractive_answer("What is the vehicle retention rate after three years?",
                                   chunks)
        assert not result.abstained
        assert "80" in result.text

    def test_never_empty_text(self):
        from src.generation import extractive_answer

        result = extractive_answer("question", [rc("a.pdf", 1, "text")])
        assert result.text.strip()