from __future__ import annotations

from pathlib import Path

import pytest

from prompting import PromptKit, PromptNotFoundError

PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def write_prompt(root: Path, folder: str, file_version: int, *, active: bool, **overrides) -> Path:
    data = {
        "name": folder,
        "version": file_version,
        "active": str(active).lower(),
        "description": "prueba",
        "system": "Sistema de prueba",
        "user": "{{ ticket }}",
        **overrides,
    }
    folder_path = root / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            f"name: {data['name']}",
            f"version: {data['version']}",
            f"active: {data['active']}",
            f"description: {data['description']}",
            "model_params:",
            "  temperature: 0",
            f"system: '{data['system']}'",
            f"user: '{data['user']}'",
        ]
    )
    path = folder_path / f"v{file_version}.yaml"
    path.write_text(body + "\n", encoding="utf-8")
    return path


@pytest.fixture
def kit() -> PromptKit:
    return PromptKit(PROMPTS_ROOT)


def test_rag_answer_v1_is_active(kit):
    prompt = kit.get("rag_answer")
    assert prompt.version == 1
    assert prompt.model_params["temperature"] == 0


def test_question_only_in_user_message(kit):
    system, user = kit.get("rag_answer").render(question="What is CONC-02?")
    assert "CONC-02" not in system["content"]
    assert user["content"] == "Question: What is CONC-02?"
    assert "<documents>" in system["content"]


def test_missing_variable_names_it(kit):
    with pytest.raises(ValueError, match="question"):
        kit.get("rag_answer").render()


def test_unknown_prompt_or_version(kit):
    with pytest.raises(PromptNotFoundError):
        kit.get("no_existe")
    with pytest.raises(PromptNotFoundError, match="v99"):
        kit.get("rag_answer", 99)


def test_two_active_versions_fail(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=True)
    write_prompt(tmp_path, "demo", 2, active=True)
    with pytest.raises(ValueError, match="more than one active version"):
        PromptKit(tmp_path).get("demo")


def test_no_active_version_fails(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=False)
    with pytest.raises(PromptNotFoundError, match="no active version"):
        PromptKit(tmp_path).get("demo")


def test_name_must_match_folder(tmp_path):
    write_prompt(tmp_path, "demo", 1, active=True, name="other_name")
    with pytest.raises(ValueError, match="folder"):
        PromptKit(tmp_path).get("demo")
