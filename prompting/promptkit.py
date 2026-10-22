"""Registry of versioned YAML prompts.

Each prompt lives at `prompts/<name>/v<N>.yaml`. Exactly one version per prompt has
`active: true`; that version is used unless a specific version is requested.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError

REQUIRED_KEYS = ("name", "version", "active", "description", "model_params", "system", "user")
_VERSION_FILE = re.compile(r"^v(\d+)\.yaml$")

# Templates load from strings, so they cannot read files through a Jinja loader.
_ENV = Environment(autoescape=False, keep_trailing_newline=False, undefined=StrictUndefined)


class PromptNotFoundError(LookupError):
    """The requested prompt or version does not exist."""


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    description: str
    model_params: dict
    system_template: str
    user_template: str
    examples: list[dict]

    def render(self, **variables) -> list[dict]:
        """Return `system` and `user` messages ready for `LLMClient.generate()`.

        `examples` is injected automatically; callers pass only their variables.
        """
        context = {"examples": self.examples, **variables}
        try:
            system = _ENV.from_string(self.system_template).render(context)
            user = _ENV.from_string(self.user_template).render(context)
        except UndefinedError as e:
            raise ValueError(
                f"Missing variable while rendering {self.name} v{self.version}: {e.message}"
            ) from e
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]


class PromptKit:
    def __init__(self, root: str | Path = "prompts") -> None:
        self.root = Path(root)

    def versions(self, name: str) -> list[int]:
        """Available prompt versions in ascending order."""
        return sorted(self._files(name))

    def get(self, name: str, version: int | None = None) -> Prompt:
        """Return the requested version, or the active version when omitted."""
        files = self._files(name)
        if version is not None:
            if version not in files:
                raise PromptNotFoundError(
                    f"{name} v{version} does not exist. Available versions: {sorted(files)}."
                )
            return self._build(name, version, self._load(files[version], name, version))

        loaded = {v: self._load(path, name, v) for v, path in files.items()}
        active = [v for v, data in loaded.items() if data["active"] is True]
        if len(active) != 1:
            paths = ", ".join(str(files[v]) for v in (active or sorted(files)))
            problem = "no active version" if not active else "more than one active version"
            error = PromptNotFoundError if not active else ValueError
            raise error(
                f"{name} has {problem}; exactly one must have `active: true`. Check: {paths}"
            )
        return self._build(name, active[0], loaded[active[0]])

    def _files(self, name: str) -> dict[int, Path]:
        folder = self.root / name
        if not folder.is_dir():
            raise PromptNotFoundError(f"Prompt {name!r} does not exist (searched {folder}).")
        files = {}
        for path in folder.glob("v*.yaml"):
            match = _VERSION_FILE.match(path.name)
            if match:
                files[int(match.group(1))] = path
        if not files:
            raise PromptNotFoundError(f"{folder} has no v<N>.yaml files.")
        return files

    @staticmethod
    def _load(path: Path, name: str, version: int) -> dict:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a YAML mapping.")
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(missing)}.")
        if data["name"] != name:
            raise ValueError(f"{path}: name is {data['name']!r}, but the folder is {name!r}.")
        if data["version"] != version:
            raise ValueError(
                f"{path}: version is {data['version']!r}, but the file is v{version}.yaml."
            )
        return data

    def _build(self, name: str, version: int, data: dict) -> Prompt:
        return Prompt(
            name=name,
            version=version,
            description=data["description"],
            model_params=dict(data["model_params"] or {}),
            system_template=data["system"],
            user_template=data["user"],
            examples=self._examples(name),
        )

    def _examples(self, name: str) -> list[dict]:
        path = self.root / name / "examples.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
