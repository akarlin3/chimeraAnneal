"""Tests for standardized AveryCorp AuditReport contract in chimeraAnneal."""
from __future__ import annotations

import json

from chimera_anneal import AuditReport, run_audit
from chimera_anneal.cli import main

REQUIRED_AUDIT_KEYS = {
    "package",
    "version",
    "suite",
    "stage_role",
    "status",
    "verdict",
    "metrics",
    "thresholds",
    "seed",
}


def test_audit_report_contract_and_determinism() -> None:
    rep1 = run_audit(seed=42)
    rep2 = run_audit(seed=42)
    assert isinstance(rep1, AuditReport)
    assert rep1 == rep2
    assert rep1.package == "chimera-anneal"
    assert rep1.suite == "Suite II: Scientific Identifiability & Null-Audit Suite"
    assert rep1.status == "PASS"
    assert "ABSORBING_HAZARD_VERIFIED" in rep1.verdict

    parsed = json.loads(rep1.to_json())
    assert set(parsed.keys()) == REQUIRED_AUDIT_KEYS
    assert parsed["seed"] == 42
    assert parsed["metrics"]["graze_gate_separated"] is True
    assert parsed["metrics"]["weibull_shape_k_mean_field"] >= 1.0
    assert "chimera-anneal" in rep1.summary_line()


def test_cli_audit_and_demo_json(capsys) -> None:
    rc_audit = main(["audit", "--seed", "42", "--json"])
    assert rc_audit == 0
    out_audit = json.loads(capsys.readouterr().out)
    assert set(out_audit.keys()) == REQUIRED_AUDIT_KEYS
    assert out_audit["package"] == "chimera-anneal"

    rc_demo = main(["demo", "--seed", "42", "--trials", "6", "--json"])
    assert rc_demo == 0
    out_demo = json.loads(capsys.readouterr().out)
    assert set(out_demo.keys()) == REQUIRED_AUDIT_KEYS
