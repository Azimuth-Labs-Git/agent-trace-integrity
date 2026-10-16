"""Everything in examples/ validates against the schemas."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def _schema(name):
    s = json.loads((ROOT / "schema" / name).read_text())
    Draft202012Validator.check_schema(s)
    return Draft202012Validator(s)


def test_expected_verdicts_validate():
    v = _schema("claim_verdict.schema.json")
    verdicts = json.loads((ROOT / "examples" / "expected_verdicts.json").read_text())
    assert verdicts
    for item in verdicts:
        v.validate(item)


def test_expected_corpus_report_validates():
    v = _schema("corpus_report.schema.json")
    v.validate(json.loads((ROOT / "examples" / "expected_corpus_report.json").read_text()))
