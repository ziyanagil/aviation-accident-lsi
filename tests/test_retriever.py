import numpy as np
import pytest

from lsi.analysis import evaluation
from lsi.engine.retriever import Retriever
from lsi.io.loader import load_cache, compute_fingerprint
from lsi.nlp.cleaner import clean_text, infer_category, parse_asn_text
from lsi.nlp.tokenizer import preprocess
from lsi.schemas import MODE_LSI, MODE_VSM, Document, EvalQuery, IndexParams

PARAMS = IndexParams(tf_mode="log", min_df=1, use_stemming=True)


@pytest.fixture(scope="module")
def documents():
    return [
        Document(
            doc_id="DOC-ENG-1",
            category="ENGINE",
            narrative="The turbine engine experienced an uncontained failure and power loss after takeoff.",
        ),
        Document(
            doc_id="DOC-ENG-2",
            category="ENGINE",
            narrative="A powerplant malfunction caused compressor surge and engine shutdown during climb.",
        ),
        Document(
            doc_id="DOC-RWY-1",
            category="RUNWAY",
            narrative="The aircraft overran the wet runway during landing rollout after heavy braking.",
        ),
        Document(
            doc_id="DOC-RWY-2",
            category="RUNWAY",
            narrative="After landing long, the jet overran the runway pavement into grass during rollout.",
        ),
        Document(
            doc_id="DOC-WX-1",
            category="WEATHER",
            narrative="Severe windshear and microburst downdraft forced a go around in heavy storm.",
        ),
        Document(
            doc_id="DOC-CFIT-1",
            category="CFIT",
            narrative="In dense cloud the crew descended below glidepath and struck mountainous terrain.",
        ),
    ]


@pytest.fixture(scope="module")
def retriever(documents):
    return Retriever.build(documents, PARAMS)


def test_matrix_dimensions(retriever, documents):
    m, n = retriever.matrix.shape
    assert n == len(documents)
    assert m == len(retriever.vectorizer.vocab)


def test_vsm_top_hit_for_exact_terms(retriever):
    hits = retriever.search("engine failure turbine", MODE_VSM, top_n=3)
    assert hits[0].category == "ENGINE"
    assert hits[0].score > 0


def test_lsi_top_hits_for_runway_query(retriever):
    hits = retriever.search("overran runway", MODE_LSI, k=3, top_n=2)
    assert {h.category for h in hits} == {"RUNWAY"}


def test_lsi_requires_k(retriever):
    with pytest.raises(ValueError):
        retriever.scores("engine", MODE_LSI)


def test_unknown_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.scores("engine", "other", 2)


def test_out_of_vocabulary_query_scores_zero(retriever):
    assert np.all(retriever.scores("zzzzqqqq", MODE_VSM) == 0)
    assert np.all(retriever.scores("zzzzqqqq", MODE_LSI, 2) == 0)


def test_compare_has_both_modes(retriever):
    frame = retriever.compare("engine failure", 2, 3)
    assert len(frame) == 3
    assert {"VSM_score", "LSI_score"} <= set(frame.columns)


def test_cache_roundtrip(tmp_path, documents):
    path = tmp_path / "cache.npz"
    first, hit_first = Retriever.load_or_build(documents, PARAMS, path)
    second, hit_second = Retriever.load_or_build(documents, PARAMS, path)
    assert (hit_first, hit_second) == (False, True)
    assert np.allclose(first.matrix, second.matrix)
    assert np.allclose(first.lsi.singular_values, second.lsi.singular_values)
    q = "engine failure"
    assert np.allclose(first.scores(q, MODE_LSI, 2), second.scores(q, MODE_LSI, 2))


def test_cache_invalidated_when_params_change(tmp_path, documents):
    path = tmp_path / "cache.npz"
    Retriever.load_or_build(documents, PARAMS, path)
    other = IndexParams(tf_mode="raw", min_df=1, use_stemming=True)
    assert load_cache(path, compute_fingerprint(documents, other)) is None


def test_evaluation_metrics():
    rel = np.array([1, 0, 1, 0])
    assert evaluation.precision_at_n(rel, 2) == 0.5
    assert evaluation.average_precision(rel) == pytest.approx((1 + 2 / 3) / 2)
    assert evaluation.reciprocal_rank(np.array([0, 0, 1])) == pytest.approx(1 / 3)
    assert evaluation.average_precision(np.zeros(3, dtype=int)) == 0.0


def test_evaluate_and_summarize(retriever):
    queries = [EvalQuery("engine failure", "ENGINE"), EvalQuery("overran runway", "RUNWAY")]
    results = evaluation.evaluate(retriever, queries, [2, 3], top_n=2)
    summary = evaluation.summarize(results, 2)
    assert set(summary["method"]) == {"VSM", "LSI"}
    assert summary["MAP"].between(0, 1).all()


def test_preprocess_pipeline():
    tokens = preprocess("The engines FAILED at 35,000 ft!", use_stemming=True)
    assert "the" not in tokens
    assert "engin" in tokens
    assert all(t.isalpha() for t in tokens)


def test_cleaner_removes_boilerplate_and_numbers():
    raw = "Aviation Safety Network\nPage 1 of 3\nThe pilot reported smoke at 12345 feet.\nhttps://asn.flightsafety.org/x"
    cleaned = clean_text(raw)
    assert "Aviation" not in cleaned
    assert "12345" not in cleaned
    assert "smoke" in cleaned


def test_parse_asn_text_and_category():
    text = ("Date: 15 JAN 2021 Time: 10:00 Type: Boeing 737-800 Owner/operator: X "
            "Phase: Landing (LDG) Narrative: The aircraft overran the runway after landing. "
            "Sources: https://asn.flightsafety.org/wikibase/1")
    parsed = parse_asn_text(text)
    assert parsed["date"] == "2021-01-15"
    assert parsed["aircraft_type"] == "Boeing 737-800"
    assert parsed["phase"] == "Landing"
    assert parsed["narrative"] == "The aircraft overran the runway after landing."
    assert infer_category(parsed["narrative"]) == "RUNWAY"
