"""Regression guard for external-memory write-back at session boundaries.

Completed turns are mirrored through MemoryManager.sync_all(), which uses a
background worker.  A short CLI/API session can otherwise invoke a provider's
on_session_end() before the final turn has reached that provider; shutdown
would cancel the queued sync and drop the whole session's write-back.
"""
from unittest.mock import MagicMock, call


def _bare_agent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    return agent


def test_shutdown_drains_turn_sync_before_provider_session_end():
    agent = _bare_agent()
    messages = [{"role": "user", "content": "preserve this completed turn"}]

    agent.shutdown_memory_provider(messages)

    assert agent._memory_manager.method_calls == [
        call.flush_pending(timeout=5.0),
        call.on_session_end(messages),
        call.shutdown_all(),
    ]


def test_session_commit_drains_turn_sync_before_provider_session_end():
    agent = _bare_agent()
    messages = [{"role": "user", "content": "commit this completed turn"}]

    agent.commit_memory_session(messages)

    assert agent._memory_manager.method_calls == [
        call.flush_pending(timeout=5.0),
        call.on_session_end(messages),
    ]
