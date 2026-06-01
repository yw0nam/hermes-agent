"""API server session lifecycle regression tests.

The API server creates ``source='api_server'`` session rows through AIAgent.
Successful one-turn requests must mark the effective session complete; otherwise
``hermes sessions list`` accumulates zombie API sessions with ``end_reason`` NULL.
"""

import asyncio

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class _FakeSessionDB:
    def __init__(self):
        self.ended = []

    def end_session(self, session_id, end_reason):
        self.ended.append((session_id, end_reason))


class _FakeAgent:
    def __init__(self, session_id="api-session", *, effective_session_id=None):
        self.session_id = effective_session_id or session_id
        self._session_db = _FakeSessionDB()
        self.session_prompt_tokens = 3
        self.session_completion_tokens = 5
        self.session_total_tokens = 8

    def run_conversation(self, *, user_message, conversation_history, task_id):
        assert task_id == "api-session"
        return {"final_response": "ok", "messages": []}


def test_run_agent_ends_effective_api_session_with_api_complete(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    fake = _FakeAgent(session_id="api-session")

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: fake)

    result, usage = asyncio.run(adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="api-session",
    ))

    assert result["final_response"] == "ok"
    assert result["session_id"] == "api-session"
    assert usage == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
    assert fake._session_db.ended == [("api-session", "api_complete")]


def test_run_agent_ends_rotated_effective_api_session(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    fake = _FakeAgent(session_id="api-session", effective_session_id="rotated-session")

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: fake)

    result, _usage = asyncio.run(adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="api-session",
    ))

    assert result["session_id"] == "rotated-session"
    assert fake._session_db.ended == [("rotated-session", "api_complete")]
