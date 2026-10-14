"""Committed configuration, policy, contracts and repository hygiene."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from agentic_core.domain.classification import DataClassification, Decision, RiskLevel
from agentic_core.models.factory import ProvidersConfig, build_providers, env_references
from agentic_core.policy import PolicyEngine, PolicyRequest, PolicySet, Resource, Subject
from agentic_core.security import SecretScanner

from context_engineering_graphrag import contracts

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ROOT / "configs" / "providers.yaml"
CONFIG_DOC = ROOT / "docs" / "operations" / "configuration.md"
POLICY = ROOT / "policies" / "context-governance.yaml"


def referenced_vars() -> set[str]:
    text = PROVIDERS.read_text(encoding="utf-8")
    return env_references(text) | set(
        re.findall(r"(?:api_key_env|access_token_env):\s*([A-Z0-9_]+)", text)
    )


def synthetic_env() -> dict[str, str]:
    env = {
        name: "1.5" if name.endswith("_PER_MTOK") else f"test-{name.lower()}"
        for name in referenced_vars()
    }
    env["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
    env["AWS_REGION"] = "us-east-1"
    env["GCP_REGION"] = "us-central1"
    return env


def test_providers_config_builds_with_resolved_env() -> None:
    config = ProvidersConfig.from_file(PROVIDERS, env=synthetic_env())
    names = [p.profile.name for p in config.providers]
    assert names[0] == "bedrock-claude"
    assert len(names) == len(set(names))
    assert len(build_providers(config, env=synthetic_env())) == len(names)


def test_providers_config_fails_closed_without_env() -> None:
    with pytest.raises(ValueError, match="unset environment variables"):
        ProvidersConfig.from_file(PROVIDERS, env={})


def test_every_variable_is_documented() -> None:
    documented = CONFIG_DOC.read_text(encoding="utf-8")
    undocumented = sorted(v for v in referenced_vars() if f"`{v}`" not in documented)
    assert undocumented == []
    settings_source = (ROOT / "src" / "context_engineering_graphrag" / "config.py").read_text(
        encoding="utf-8"
    )
    fields = re.findall(r"^    ([a-z_]+): ", settings_source, re.M)
    missing = sorted(f for f in fields if f"`GRAPHRAG_{f.upper()}`" not in documented)
    assert missing == []


def decide(engine: PolicyEngine, **kw: object) -> Decision:
    values: dict[str, object] = {
        "subject": Subject(id="a", attributes={"status": "active", "action_in_scope": True}),
        "action": "rag:query",
        "resource": Resource(type="collection", classification=DataClassification.CONFIDENTIAL),
    }
    values.update(kw)
    return engine.evaluate(PolicyRequest.model_validate(values)).decision


def test_access_policy_behaviour() -> None:
    engine = PolicyEngine(PolicySet.from_file(POLICY))
    restricted = Resource(type="collection", classification=DataClassification.RESTRICTED)
    assert decide(engine) is Decision.ALLOW
    assert decide(engine, resource=restricted) is Decision.DENY
    assert (
        decide(engine, resource=restricted, environment={"purpose_of_use": "TREAT"})
        is Decision.ALLOW
    )
    assert decide(engine, risk=RiskLevel.CRITICAL) is Decision.REQUIRE_APPROVAL
    out_of_scope = Subject(id="a", attributes={"status": "active", "action_in_scope": False})
    assert decide(engine, subject=out_of_scope) is Decision.DENY
    delegated = Subject(
        id="a", attributes={"status": "active", "action_in_scope": True, "delegated": True}
    )
    assert (
        decide(engine, subject=delegated, action="rag:document.delete") is Decision.REQUIRE_APPROVAL
    )


@pytest.mark.parametrize(
    ("attributes", "classification", "expected"),
    [
        (
            {
                "groundedness": 0.9,
                "abstained": False,
                "contradicted_claims": 0,
                "invalid_citations": 0,
            },
            2,
            Decision.ALLOW,
        ),
        (
            {
                "groundedness": 0.3,
                "abstained": False,
                "contradicted_claims": 0,
                "invalid_citations": 0,
            },
            2,
            Decision.DENY,
        ),
        (
            {
                "groundedness": 1.0,
                "abstained": True,
                "contradicted_claims": 0,
                "invalid_citations": 0,
            },
            3,
            Decision.ALLOW,
        ),
        (
            {
                "groundedness": 0.9,
                "abstained": False,
                "contradicted_claims": 1,
                "invalid_citations": 0,
            },
            2,
            Decision.DENY,
        ),
        (
            {
                "groundedness": 0.9,
                "abstained": False,
                "contradicted_claims": 0,
                "invalid_citations": 2,
            },
            2,
            Decision.DENY,
        ),
        (
            {
                "groundedness": 0.7,
                "abstained": False,
                "contradicted_claims": 0,
                "invalid_citations": 0,
            },
            3,
            Decision.REQUIRE_APPROVAL,
        ),
    ],
)
def test_release_policy_behaviour(
    attributes: dict[str, object], classification: int, expected: Decision
) -> None:
    engine = PolicyEngine(PolicySet.from_file(POLICY))
    resource = Resource(
        type="answer",
        id="c",
        classification=DataClassification(classification),
        attributes=attributes,
    )
    subject = Subject(id="a", attributes={"delegated": False})
    assert decide(engine, subject=subject, action="answer:release", resource=resource) is expected


def test_contracts_are_current(tmp_path: Path) -> None:
    assert contracts.main(["--check"], root=ROOT) == 0
    assert contracts.main(["--check"], root=tmp_path) == 1
    assert contracts.main([], root=tmp_path) == 0
    assert contracts.main(["--check"], root=tmp_path) == 0
    assert (tmp_path / "docs" / "api" / "openapi.json").exists()


def test_no_placeholders_in_committed_files() -> None:
    pattern = re.compile(r"YOUR[-_]|REPLACE[-_ ]ME|CHANGEME|<your|\bTODO\b|\bFIXME\b", re.I)
    offenders = [
        str(p.relative_to(ROOT))
        for folder in ("configs", "policies", "docs", "src", "deployments")
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
        and p.suffix in {".yaml", ".yml", ".md", ".py", ".tf", ".tpl", ".json"}
        and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


HCL_REFERENCE = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_*\[\]]+)+")


def test_no_credentials_in_committed_files() -> None:
    scanner = SecretScanner()
    offenders = []
    for folder in ("configs", "policies", "docs", "src", "deployments", ".github"):
        for p in (ROOT / folder).rglob("*"):
            if not p.is_file() or p.suffix not in {".yaml", ".yml", ".md", ".py", ".toml", ".tf"}:
                continue
            text = p.read_text(encoding="utf-8")
            for f in scanner.detect(text):
                value = text[f.start : f.end]
                # An unquoted Terraform reference (random_password.db.result) is not a literal.
                if (
                    p.suffix == ".tf"
                    and f.kind == "generic_secret"
                    and HCL_REFERENCE.fullmatch(value)
                ):
                    continue
                offenders.append(f"{p.relative_to(ROOT)}:{f.kind}")
    assert offenders == []
