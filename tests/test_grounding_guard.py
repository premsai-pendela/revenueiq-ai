"""Unit tests for the LLM grounding guard — the feature that verifies every
number in the AI report traces back to real source data."""
from src.llm.grounded_report import extract_numbers, verify, Fact, _to_float


def test_to_float_handles_suffixes_and_commas():
    assert _to_float("9.87", "M") == 9.87e6
    assert _to_float("320", "K") == 320_000
    assert _to_float("1,234") == 1234.0


def test_extract_detects_currency_percent_and_counts():
    kinds = {e.kind for e in extract_numbers("Revenue $9.87M, churn 48.8%, 707 customers.")}
    assert {"currency", "percent", "count"} <= kinds


def test_small_integers_and_years_are_structural_not_flagged():
    # "3 models" and the year 2011 must not be treated as business claims
    kinds = [e.value for e in extract_numbers("In 2011 we built 3 models.")]
    assert 2011 not in kinds and 3 not in kinds


def test_verify_grounds_matching_numbers():
    facts = [Fact("rev", "revenue", 9_865_267.97, "currency"),
             Fact("churn", "churn rate", 48.8, "percent"),
             Fact("cust", "customers", 4372, "count")]
    audit = verify("Revenue is $9.87M, churn 48.8%, across 4,372 customers.", facts)
    assert audit["ungrounded"] == 0
    assert audit["grounded"] == audit["checked"] == 3
    assert audit["grounding_score"] == 1.0


def test_verify_flags_a_hallucinated_number():
    facts = [Fact("rev", "revenue", 9_865_267.97, "currency")]
    # the LLM invented a "95%" that exists nowhere in the facts
    audit = verify("Revenue is $9.87M with a 95% retention rate.", facts)
    assert audit["ungrounded"] >= 1


def test_verify_respects_rounding_tolerance():
    facts = [Fact("rev", "revenue", 5_537_125.24, "currency")]
    # "$5.54M" rounds to 5,540,000 — within tolerance, should ground
    assert verify("The tier holds $5.54M.", facts)["ungrounded"] == 0
