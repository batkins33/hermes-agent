from __future__ import annotations

import json

from hermes_cli import paperclip_identity_probe as probe


def test_build_receipt_reports_only_safe_child_identity_claims(monkeypatch):
    responses = iter(
        [
            probe.JsonResponse(status=200, payload={"id": "agent-1", "name": "TF Hermes Lead"}),
            probe.JsonResponse(status=200, payload={"id": "run-1", "agentId": "agent-1", "status": "running"}),
        ],
    )
    monkeypatch.setattr(probe, "_read_json", lambda *_args: next(responses))
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-1")
    monkeypatch.setenv("PAPERCLIP_TASK_ID", "issue-1")
    monkeypatch.setenv("PAPERCLIP_API_KEY", "never-render-this-value")
    monkeypatch.delenv("PAPERCLIP_AGENT_JWT_SECRET", raising=False)

    receipt = probe.build_receipt(
        api_url="http://127.0.0.1:3100/api",
        expected_agent_id="agent-1",
        expected_issue_id="issue-1",
        expected_issue_identifier="PAP-2367",
    )

    assert receipt == {
        "version": 1,
        "execution_mode": "fresh_non_resumed",
        "child_paperclip_run_id": "run-1",
        "selected_issue_id": "issue-1",
        "selected_issue_identifier": "PAP-2367",
        "expected_issue_id": "issue-1",
        "identity_api_status": 200,
        "identity_name": "TF Hermes Lead",
        "identity_id": "agent-1",
        "expected_agent_id": "agent-1",
        "heartbeat_api_status": 200,
        "heartbeat_run_id": "run-1",
        "heartbeat_agent_id": "agent-1",
        "heartbeat_run_matches_child": True,
        "signing_secret_present": False,
    }
    assert "never-render-this-value" not in json.dumps(receipt)


def test_build_receipt_reports_signing_secret_presence_without_its_value(monkeypatch):
    monkeypatch.setattr(probe, "_read_json", lambda *_args: probe.JsonResponse(status=0, payload=None))
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-1")
    monkeypatch.setenv("PAPERCLIP_TASK_ID", "issue-1")
    monkeypatch.setenv("PAPERCLIP_API_KEY", "never-render-this-value")
    monkeypatch.setenv("PAPERCLIP_AGENT_JWT_SECRET", "never-render-this-value-either")

    receipt = probe.build_receipt(
        api_url="http://127.0.0.1:3100/api",
        expected_agent_id="agent-1",
        expected_issue_id="issue-1",
        expected_issue_identifier="PAP-2367",
    )

    assert receipt["signing_secret_present"] is True
    serialized = json.dumps(receipt)
    assert "never-render-this-value" not in serialized
