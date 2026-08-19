"""Regression tests for browser tab affinity correction."""

import json
import threading

from tools import browser_tool


def _base_nav_patches(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_poisoned_task_sessions", {})
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda key: {"_first_nav": True})
    monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda key: None)
    monkeypatch.setattr(browser_tool, "_get_open_command_timeout", lambda first_open=False: 1)
    monkeypatch.setattr(browser_tool, "_copy_fallback_warning", lambda response, result: response)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
    monkeypatch.setattr(browser_tool, "_sensitive_query_param_name", lambda url: None)


def test_reused_cdp_daemon_switches_to_matching_tab(monkeypatch):
    _base_nav_patches(monkeypatch)
    calls = []

    def fake_run(task_id, command, args=None, timeout=None, _engine_override=None):
        calls.append((task_id, command, args, timeout, _engine_override))
        if command == "open":
            return {"success": True, "data": {"title": "Example Domain", "url": "https://example.com/"}}
        if command == "eval":
            return {"success": True, "data": {"result": "https://www.google.com/" if len([c for c in calls if c[1] == 'eval']) == 1 else "https://example.com/"}}
        if command == "tab list":
            return {
                "success": True,
                "data": {
                    "tabs": [
                        {"tabId": "t1", "url": "https://example.com/"},
                        {"tabId": "t2", "url": "https://www.google.com/"},
                    ]
                },
            }
        if command == "tab":
            assert args == ["t1"]
            return {"success": True}
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "example", "refs": {}}}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    result = json.loads(browser_tool.browser_navigate("https://example.com", task_id="task-1"))
    assert result["success"] is True
    assert result["url"] == "https://example.com/"
    assert result["tab_affinity"] == "switched"
    assert [c[1] for c in calls] == ["open", "eval", "tab list", "tab", "eval", "snapshot"]


def test_affinity_short_circuits_when_active_url_already_matches(monkeypatch):
    _base_nav_patches(monkeypatch)
    calls = []

    def fake_run(task_id, command, args=None, timeout=None, _engine_override=None):
        calls.append(command)
        if command == "open":
            return {"success": True, "data": {"title": "Example Domain", "url": "https://example.com/"}}
        if command == "eval":
            return {"success": True, "data": {"result": "https://example.com"}}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    result = json.loads(browser_tool.browser_navigate("https://example.com", task_id="task-1"))
    assert result["success"] is True
    assert "tab_affinity" not in result
    assert calls == ["open", "eval", "snapshot"]


def test_no_matching_tab_returns_safe_failure(monkeypatch):
    _base_nav_patches(monkeypatch)

    def fake_run(task_id, command, args=None, timeout=None, _engine_override=None):
        if command == "open":
            return {"success": True, "data": {"title": "Example Domain", "url": "https://example.com/"}}
        if command == "eval":
            return {"success": True, "data": {"result": "https://www.google.com/"}}
        if command == "tab list":
            return {"success": True, "data": {"tabs": [{"tabId": "t2", "url": "https://www.google.com/"}]}}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    result = json.loads(browser_tool.browser_navigate("https://example.com", task_id="task-1"))
    assert result["success"] is False
    assert "snapshot" not in result
    assert "affinity" in result["error"].lower()
    assert browser_tool._poisoned_task_sessions == {"task-1": "tab_affinity_failed"}
    follow_up = json.loads(browser_tool.browser_snapshot(task_id="task-1"))
    assert follow_up["success"] is False
    assert "navigate again" in follow_up["error"]


def test_malformed_tab_list_or_failed_switch_returns_safe_failure(monkeypatch):
    _base_nav_patches(monkeypatch)

    def fake_run(task_id, command, args=None, timeout=None, _engine_override=None):
        if command == "open":
            return {"success": True, "data": {"title": "Example Domain", "url": "https://example.com/"}}
        if command == "eval":
            return {"success": True, "data": {"result": "https://www.google.com/"}}
        if command == "tab list":
            return {"success": True, "data": {"tabs": "broken"}}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    result = json.loads(browser_tool.browser_navigate("https://example.com", task_id="task-1"))
    assert result["success"] is False
    assert "snapshot" not in result


def test_local_behavior_unchanged(monkeypatch):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_navigation_session_key", lambda task_id, url: task_id)
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_poisoned_task_sessions", {})
    monkeypatch.setattr(browser_tool, "_get_session_info", lambda key: {"_first_nav": True})
    monkeypatch.setattr(browser_tool, "_maybe_start_recording", lambda key: None)
    monkeypatch.setattr(browser_tool, "_get_open_command_timeout", lambda first_open=False: 1)
    monkeypatch.setattr(browser_tool, "_copy_fallback_warning", lambda response, result: response)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: True)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: True)
    monkeypatch.setattr(browser_tool, "_sensitive_query_param_name", lambda url: None)

    calls = []

    def fake_run(task_id, command, args=None, timeout=None, _engine_override=None):
        calls.append(command)
        if command == "open":
            return {"success": True, "data": {"title": "Example Domain", "url": "https://example.com/"}}
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "ok", "refs": {}}}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)

    result = json.loads(browser_tool.browser_navigate("https://example.com", task_id="task-1"))
    assert result["success"] is True
    assert calls == ["open", "snapshot"]


def test_url_normalization_is_narrow():
    assert browser_tool._normalize_url_for_tab_affinity("https://a.test/path/") == "https://a.test/path"
    assert browser_tool._normalize_url_for_tab_affinity("https://a.test/path") == "https://a.test/path"
    assert browser_tool._normalize_url_for_tab_affinity("https://a.test/other") != browser_tool._normalize_url_for_tab_affinity("https://a.test/path")
    assert browser_tool._normalize_url_for_tab_affinity("https://b.test/path") != browser_tool._normalize_url_for_tab_affinity("https://a.test/path")


def test_affinity_failure_restores_distinct_owned_previous_binding(monkeypatch):
    _base_nav_patches(monkeypatch)
    previous_key = "task-1::previous"
    browser_tool._last_active_session_key["task-1"] = previous_key
    monkeypatch.setattr(browser_tool, "_active_sessions", {
        previous_key: {"owner_task_id": "task-1", "session_key": previous_key},
    })

    def fake_run(task_id, command, args=None, timeout=None, _engine_override=None):
        if command == "open":
            return {"success": True, "data": {"title": "Example Domain", "url": "https://example.com/"}}
        if command == "eval":
            return {"success": True, "data": {"result": "https://www.google.com/"}}
        if command == "tab list":
            return {"success": True, "data": {"tabs": []}}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
    result = json.loads(browser_tool.browser_navigate("https://example.com", task_id="task-1"))

    assert result["success"] is False
    assert browser_tool._last_active_session_key["task-1"] == previous_key
    assert browser_tool._poisoned_task_sessions == {}


def test_successful_navigation_clears_prior_poison(monkeypatch):
    _base_nav_patches(monkeypatch)
    browser_tool._poisoned_task_sessions["task-1"] = "tab_affinity_failed"
    concurrent_results = []

    def fake_run(task_id, command, args=None, timeout=None, _engine_override=None):
        if command == "open":
            return {"success": True, "data": {"title": "Example Domain", "url": "https://example.com/"}}
        if command == "eval":
            worker = threading.Thread(
                target=lambda: concurrent_results.append(
                    json.loads(browser_tool.browser_snapshot(task_id="task-1"))
                )
            )
            worker.start()
            worker.join(timeout=2)
            assert not worker.is_alive()
            return {"success": True, "data": {"result": "https://example.com/"}}
        if command == "snapshot":
            return {"success": True, "data": {"snapshot": "example", "refs": {}}}
        raise AssertionError(command)

    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run)
    result = json.loads(browser_tool.browser_navigate("https://example.com", task_id="task-1"))

    assert result["success"] is True
    assert concurrent_results
    assert all(item["success"] is False for item in concurrent_results)
    assert all("navigate again" in item["error"] for item in concurrent_results)
    assert browser_tool._poisoned_task_sessions == {}
