"""Original-side Codex CLI agent; no Harbor or adapters dependency.

Like EvoEval's original-side agent, this runner supplies a workspace, invokes
Codex, and reads its answer file. The benchmark's scorer remains authoritative.
"""

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import tempfile

from ..data import Sample

logger = logging.getLogger(__name__)

# Reuse this exact suffix on the Harbor side when running parity experiments.
ANSWER_INSTRUCTION = (
    "Solve the question using the provided attachments and available tools.\n"
    "Write your complete final answer to answer.txt in the working directory.\n"
    "Follow the question's requested format. For code-generation questions, "
    "include the complete executable code in answer.txt, not just a path to it.\n"
    "Do not ask for human assistance."
)


def _redact_config(value: str) -> str:
    key, _, _ = value.partition("=")
    if any(word in key.lower() for word in ("key", "token", "secret", "password", "authorization")):
        return key + "=<redacted>"
    return value


class CodexAgent:
    def __init__(
        self,
        *,
        model: str,
        files_dir: str | pathlib.Path,
        logs_dir: str | pathlib.Path,
        codex_bin: str = "codex",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 3600,
        max_attempts: int = 1,
        attach_files: bool = True,
        reasoning_effort: str | None = None,
        web_search: str = "disabled",
        config_overrides: list[str] | None = None,
        full_access: bool = False,
    ) -> None:
        executable = shutil.which(codex_bin)
        if executable is None:
            raise ValueError("Codex CLI not found. Install @openai/codex or pass --codex-bin /path/to/codex.")
        try:
            version = subprocess.run(
                [executable, "--version"], capture_output=True, text=True, timeout=15, check=True
            ).stdout.strip()
        except subprocess.SubprocessError as exc:
            raise RuntimeError(f"Cannot run {codex_bin} --version: {exc}") from exc
        if timeout <= 0 or max_attempts < 1:
            raise ValueError("Codex timeout and max_attempts must be positive.")
        if any(not item.partition("=")[0].strip() or "=" not in item for item in config_overrides or []):
            raise ValueError("Each --codex-config must have the form KEY=VALUE.")
        self.executable = str(pathlib.Path(executable).absolute())
        self.model = model
        self.files_dir = pathlib.Path(files_dir).resolve()
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.attach_files = attach_files
        self.full_access = full_access
        self.config = ['approval_policy="never"', f"web_search={json.dumps(web_search)}"]
        if reasoning_effort is not None:
            self.config.append(f"model_reasoning_effort={json.dumps(reasoning_effort)}")

        self.env = os.environ.copy()
        # Explicit benchmark connection settings must override a user's login
        # provider. Credentials travel only through the subprocess environment.
        if api_key or base_url:
            key = api_key or self.env.get("CODEX_API_KEY") or self.env.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("A custom Codex endpoint requires --api-key, PREDICT_LLM_API_KEY, OPENAI_API_KEY, or CODEX_API_KEY.")
            self.env["CODEX_API_KEY"] = key
            self.env["OPENAI_API_KEY"] = key
            self.config.extend([
                'model_provider="physcibench"',
                'model_providers.physcibench.name="PhySciBench"',
                f"model_providers.physcibench.base_url={json.dumps(base_url or 'https://api.openai.com/v1')}",
                'model_providers.physcibench.wire_api="responses"',
                'model_providers.physcibench.env_key="CODEX_API_KEY"',
            ])
        self.config.extend(config_overrides or [])
        logs_root = pathlib.Path(logs_dir).resolve()
        logs_root.mkdir(parents=True, exist_ok=True)
        self.logs_dir = pathlib.Path(tempfile.mkdtemp(prefix="run-", dir=logs_root))
        metadata = {
            "agent": "codex", "version": version, "model": model,
            "timeout": timeout, "max_attempts": max_attempts,
            "attach_files": attach_files, "full_access": full_access,
            "config": [_redact_config(value) for value in self.config],
            "note": "Codex also loads the user's existing config/auth; align it on both parity sides.",
        }
        (self.logs_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Codex %s; logs/workspaces: %s", version, self.logs_dir)

    def _prepare_workspace(self, sample: Sample, workspace: pathlib.Path) -> str:
        """Copy only referenced inputs, never the benchmark answer/rubrics."""
        attachments = []
        if self.attach_files:
            for filename in sample.files:
                if filename.startswith(("http://", "https://")):
                    attachments.append(filename)
                    continue
                relative = pathlib.Path(filename)
                source = (self.files_dir / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(self.files_dir):
                    raise ValueError(f"Unsafe attachment path: {filename}")
                if not source.is_file():
                    raise FileNotFoundError(f"Missing attachment: {filename}; run scripts/download.py --with-files.")
                target = workspace / "files" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                attachments.append("files/" + relative.as_posix())
        prompt = sample.raw_question
        if attachments:
            prompt += "\n\nAttachments (paths are relative to the working directory):\n"
            prompt += "\n".join(f"- {name}" for name in attachments)
        return prompt + "\n\n" + ANSWER_INSTRUCTION + "\n"

    async def _run(self, prompt: str, attempt_dir: pathlib.Path, workspace: pathlib.Path) -> str:
        command = [
            self.executable, "exec", "--skip-git-repo-check", "--model", self.model,
            "--json", "--output-last-message", str(attempt_dir / "final_message.txt"),
        ]
        if self.full_access:
            command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            command.extend(["--sandbox", "workspace-write"])
        for value in self.config:
            command.extend(["-c", value])
        command.append("-")
        with (attempt_dir / "events.jsonl").open("wb") as stdout, (attempt_dir / "stderr.log").open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=workspace, env=self.env, stdin=asyncio.subprocess.PIPE,
                stdout=stdout, stderr=stderr, start_new_session=(os.name == "posix"),
            )
            try:
                await asyncio.wait_for(process.communicate(prompt.encode("utf-8")), timeout=self.timeout)
            except BaseException:
                # Stop shell/tool descendants as well as the CLI on timeout or cancellation.
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    elif process.returncode is None:
                        process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
                raise
        if process.returncode:
            raise RuntimeError(f"Codex exited with code {process.returncode}; see {attempt_dir / 'stderr.log'}")
        answer = workspace / "answer.txt"
        if answer.is_symlink() or not answer.is_file():
            raise RuntimeError("Codex did not produce answer.txt; see the saved events/final_message.")
        response = answer.read_text(encoding="utf-8")
        if not response.strip():
            raise RuntimeError("Codex produced an empty answer.txt.")
        return response

    async def predict(self, sample: Sample) -> str | None:
        digest = hashlib.sha256(sample.id.encode()).hexdigest()[:8]
        sample_name = re.sub(r"[^A-Za-z0-9_.-]", "_", sample.id)[:80] + "-" + digest
        for attempt in range(1, self.max_attempts + 1):
            attempt_dir = self.logs_dir / sample_name / f"attempt-{attempt}"
            workspace = attempt_dir / "workspace"
            workspace.mkdir(parents=True)
            try:
                prompt = self._prepare_workspace(sample, workspace)
                (attempt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                return await self._run(prompt, attempt_dir, workspace)
            except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                (attempt_dir / "error.txt").write_text(message, encoding="utf-8")
                logger.warning("Sample %s: Codex attempt %d/%d failed: %s", sample.id, attempt, self.max_attempts, message)
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 30))
        return None
