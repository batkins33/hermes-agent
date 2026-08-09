"""Deterministic, safe identity receipt for a Paperclip-launched Hermes child."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROOF_PREFIX = "PAPERCLIP_IDENTITY_PROOF_JSON="


@dataclass(frozen=True)
class JsonResponse:
    status: int
    payload: dict[str, Any] | None


def _read_json(url: str, api_key: str) -> JsonResponse:
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - explicit Paperclip control-plane URL
            raw = response.read()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            return JsonResponse(status=response.status, payload=payload if isinstance(payload, dict) else None)
    except HTTPError as error:
        return JsonResponse(status=error.code, payload=None)
    except (URLError, TimeoutError, OSError):
        return JsonResponse(status=0, payload=None)


def build_receipt(*, api_url: str, expected_agent_id: str, expected_issue_id: str, expected_issue_identifier: str) -> dict[str, Any]:
    """Gather only non-secret facts from the already-sanitized child environment."""
    run_id = os.environ.get("PAPERCLIP_RUN_ID", "")
    selected_issue_id = os.environ.get("PAPERCLIP_TASK_ID", "")
    api_key = os.environ.get("PAPERCLIP_API_KEY", "")
    base_url = api_url.rstrip("/")

    identity = _read_json(f"{base_url}/agents/me", api_key) if api_key else JsonResponse(status=0, payload=None)
    heartbeat = (
        _read_json(f"{base_url}/heartbeat-runs/{run_id}", api_key)
        if api_key and run_id
        else JsonResponse(status=0, payload=None)
    )
    identity_payload = identity.payload or {}
    heartbeat_payload = heartbeat.payload or {}
    heartbeat_run_id = heartbeat_payload.get("id") if isinstance(heartbeat_payload.get("id"), str) else None
    heartbeat_agent_id = heartbeat_payload.get("agentId") if isinstance(heartbeat_payload.get("agentId"), str) else None

    return {
        "version": 1,
        "execution_mode": "fresh_non_resumed",
        "child_paperclip_run_id": run_id or None,
        "selected_issue_id": selected_issue_id or None,
        "selected_issue_identifier": expected_issue_identifier,
        "expected_issue_id": expected_issue_id,
        "identity_api_status": identity.status,
        "identity_name": identity_payload.get("name") if isinstance(identity_payload.get("name"), str) else None,
        "identity_id": identity_payload.get("id") if isinstance(identity_payload.get("id"), str) else None,
        "expected_agent_id": expected_agent_id,
        "heartbeat_api_status": heartbeat.status,
        "heartbeat_run_id": heartbeat_run_id,
        "heartbeat_agent_id": heartbeat_agent_id,
        "heartbeat_run_matches_child": heartbeat_run_id == run_id and bool(run_id),
        "signing_secret_present": "PAPERCLIP_AGENT_JWT_SECRET" in os.environ,
    }


def run(args: Any) -> int:
    receipt = build_receipt(
        api_url=args.api_url,
        expected_agent_id=args.expected_agent_id,
        expected_issue_id=args.expected_issue_id,
        expected_issue_identifier=args.expected_issue_identifier,
    )
    # The receipt deliberately excludes API keys, JWTs, secret values, and raw HTTP bodies.
    print(f"{PROOF_PREFIX}{json.dumps(receipt, separators=(',', ':'), sort_keys=True)}")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Emit a safe Paperclip child identity receipt.")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--expected-agent-id", required=True)
    parser.add_argument("--expected-issue-id", required=True)
    parser.add_argument("--expected-issue-identifier", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
