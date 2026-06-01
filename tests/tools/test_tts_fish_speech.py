"""Tests for the built-in Fish-Speech TTS provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import tts_tool


class _FakeResponse:
    def __init__(self, content: bytes = b"RIFFfakewav") -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_generate_fish_speech_posts_openai_compatible_payload(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirrors requests.post
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    out = tmp_path / "speech.wav"

    result = tts_tool._generate_fish_speech_tts(
        "hello fish",
        str(out),
        {"fish-speech": {"voice": "default"}},
    )

    assert result == str(out)
    assert out.read_bytes() == b"RIFFfakewav"
    assert calls == [
        {
            "url": "http://192.168.0.41:8092/v1/audio/speech",
            "headers": {"Content-Type": "application/json"},
            "json": {"input": "hello fish", "voice": "default", "response_format": "wav"},
            "timeout": 120.0,
        }
    ]


def test_generate_fish_speech_passes_voice_clone_fields(monkeypatch, tmp_path):
    payloads = []

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirrors requests.post
        payloads.append(json)
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    tts_tool._generate_fish_speech_tts(
        "cloned voice",
        str(tmp_path / "speech.wav"),
        {
            "fish-speech": {
                "ref_audio": "https://example.com/reference.wav",
                "ref_text": "reference transcript",
            }
        },
    )

    assert payloads[0]["ref_audio"] == "https://example.com/reference.wav"
    assert payloads[0]["ref_text"] == "reference transcript"


def test_generate_fish_speech_resolves_named_reference_voice(monkeypatch, tmp_path):
    payloads = []
    voice_dir = tmp_path / "voices" / "natsume"
    voice_dir.mkdir(parents=True)
    (voice_dir / "merged_audio.mp3").write_bytes(b"audio")
    (voice_dir / "combined.lab").write_text("reference transcript", encoding="utf-8")

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirrors requests.post
        payloads.append(json)
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    tts_tool._generate_fish_speech_tts(
        "named cloned voice",
        str(tmp_path / "speech.wav"),
        {
            "fish-speech": {
                "reference_voice": "natsume",
                "reference_voices_dir": str(tmp_path / "voices"),
            }
        },
    )

    assert payloads[0]["ref_audio"].startswith("data:audio/mpeg;base64,")
    assert payloads[0]["ref_text"] == "reference transcript"


def test_fish_speech_reference_voice_picker_writes_ref_paths(monkeypatch, tmp_path):
    from hermes_cli import tools_config

    voices_root = tmp_path / "voices"
    first = voices_root / "alpha"
    second = voices_root / "natsume"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    for voice_dir in (first, second):
        (voice_dir / "merged_audio.mp3").write_bytes(b"audio")
        (voice_dir / "combined.lab").write_text("transcript", encoding="utf-8")

    monkeypatch.setattr(tools_config, "_prompt_choice", lambda question, choices, default=0: 2)
    config = {"tts": {"fish-speech": {"reference_voices_dir": str(voices_root)}}}

    getattr(tools_config, "_configure_fish_speech_reference_voice")(config)

    fish_cfg = config["tts"]["fish-speech"]
    assert fish_cfg["reference_voice"] == "natsume"
    assert fish_cfg["ref_audio"] == str(second / "merged_audio.mp3")
    assert fish_cfg["ref_text"] == str(second / "combined.lab")


def test_fish_speech_provider_uses_wav_default_path(monkeypatch, tmp_path):
    generated_paths = []

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "fish-speech"})
    monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(tmp_path))

    def fake_generate(text, output_path, tts_config):
        generated_paths.append(Path(output_path))
        Path(output_path).write_bytes(b"RIFFfakewav")
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_fish_speech_tts", fake_generate)

    result = tts_tool.json.loads(tts_tool.text_to_speech_tool("hello fish"))

    assert result["success"] is True
    assert result["provider"] == "fish-speech"
    assert generated_paths[0].suffix == ".wav"


def test_fish_speech_is_builtin_and_in_picker():
    from hermes_cli import tools_config

    assert "fish-speech" in tts_tool.BUILTIN_TTS_PROVIDERS
    tts_cat = tools_config.TOOL_CATEGORIES["tts"]
    visible = tools_config._visible_providers(tts_cat, config={})
    fish_rows = [row for row in visible if row.get("tts_provider") == "fish-speech"]

    assert len(fish_rows) == 1
    assert fish_rows[0]["name"] == "Fish-Speech"
