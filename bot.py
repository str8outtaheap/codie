import asyncio
import json
import os
import posixpath
import re
import shlex
import textwrap
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CODEX_CMD = os.getenv("CODEX_CMD", "codex")
DEFAULT_WORKDIR = os.path.abspath(os.getenv("CODEX_WORKDIR") or os.getcwd())
VOICE_TRANSCRIPTION = os.getenv("VOICE_TRANSCRIPTION", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
)

MAX_TELEGRAM_CHUNK = 4096
PROGRESS_MAX_ACTIONS = 5
PROGRESS_COMMAND_WIDTH = 200
PROGRESS_EDIT_MIN_INTERVAL_S = 1.0
PROGRESS_TICK_S = 3.0
OPENAI_AUDIO_MAX_BYTES = 25 * 1024 * 1024
OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
STATE_DIR = os.path.join(os.path.expanduser("~"), ".codie")
STATE_PATH = os.path.join(STATE_DIR, "state.json")
PROJECTS_PATH = os.path.join(STATE_DIR, "projects.json")
MACHINES_PATH = os.path.join(STATE_DIR, "machines.toml")

SYSTEM_HINT = (
    "You are Codex running in a Telegram bridge. "
    "Tools are enabled; use them directly for commands and edits. "
    "Answer directly and concisely. "
    "You have filesystem access; run commands yourself when needed. "
    "Only include file paths where helpful; don't tack them onto every line."
)

RESUME_RE = re.compile(r"(?im)^\s*`?codex\s+resume\s+(?P<token>[^`\s]+)`?\s*$")

# Data models
@dataclass
class BotState:
    workdir: str = DEFAULT_WORKDIR
    has_session: bool = False
    resume_token: str | None = None
    resume_map: dict[str, str] = field(default_factory=dict)
    project_map: dict[str, str] = field(default_factory=dict)
    machines: dict[str, dict[str, str]] = field(default_factory=dict)
    active_machine: str = "local"
    machine_workdir: dict[str, str] = field(default_factory=dict)
    pin: str | None = None
    run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class CodexRunResult:
    message: str | None
    resume: str | None
    error: str | None
    exit_code: int | None


@dataclass(slots=True)
class ProgressUpdate:
    action_id: str
    kind: str
    title: str
    phase: str
    ok: bool | None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionState:
    action_id: str
    kind: str
    title: str
    phase: str
    ok: bool | None
    detail: dict[str, Any]
    display_phase: str
    completed: bool
    first_seen: int
    last_update: int


class ProgressTracker:
    def __init__(self) -> None:
        self.action_count = 0
        self._actions: dict[str, ActionState] = {}
        self._seq = 0

    def note(self, update: ProgressUpdate) -> bool:
        action_id = update.action_id
        if not action_id:
            return False
        completed = update.phase == "completed"
        existing = self._actions.get(action_id)
        has_open = existing is not None and not existing.completed
        is_update = update.phase == "updated" or (update.phase == "started" and has_open)
        display_phase = "updated" if is_update and not completed else update.phase

        self._seq += 1
        seq = self._seq
        if existing is None:
            self.action_count += 1
            first_seen = seq
        else:
            first_seen = existing.first_seen
        self._actions[action_id] = ActionState(
            action_id=action_id,
            kind=update.kind,
            title=update.title,
            phase=update.phase,
            ok=update.ok,
            detail=update.detail,
            display_phase=display_phase,
            completed=completed,
            first_seen=first_seen,
            last_update=seq,
        )
        return True

    def snapshot(self) -> list[ActionState]:
        return sorted(self._actions.values(), key=lambda item: item.first_seen)


# Progress rendering
def format_elapsed(elapsed_s: float) -> str:
    total = max(0, int(elapsed_s))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def shorten(text: str, width: int | None) -> str:
    if width is None:
        return text
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return textwrap.shorten(text, width=width, placeholder="...")


def sanitize_code(text: str) -> str:
    return text.replace("`", "'")


def _format_change_summary(changes: list[dict[str, str]]) -> str:
    if not changes:
        return "files changed"
    return f"files changed ({len(changes)})"


def format_action_title(action: ActionState) -> str:
    title = action.title or ""
    if action.kind == "command":
        title = shorten(title, PROGRESS_COMMAND_WIDTH)
        return f"cmd: `{sanitize_code(title)}`"
    if action.kind == "tool":
        title = shorten(title, PROGRESS_COMMAND_WIDTH)
        return f"tool: `{sanitize_code(title)}`"
    if action.kind == "file_change":
        changes = action.detail.get("changes")
        if isinstance(changes, list):
            return _format_change_summary(changes)
        return "files changed"
    title = shorten(title, PROGRESS_COMMAND_WIDTH)
    return f"{action.kind}: `{sanitize_code(title)}`"


def format_action_line(action: ActionState) -> str:
    if action.display_phase != "completed":
        status = "UPD" if action.display_phase == "updated" else "RUN"
    else:
        ok = action.ok
        if ok is None or ok is True:
            status = "OK"
        else:
            status = "ERR"
    return f"{status} {format_action_title(action)}"


def render_progress(
    tracker: ProgressTracker,
    *,
    elapsed_s: float,
    label: str,
    pin: str | None = None,
    host: str | None = None,
) -> str:
    header = f"{label} {format_elapsed(elapsed_s)}"
    lines: list[str] = [header]
    if host:
        lines.append(f"host: {host}")
    if pin:
        lines.append(f"pin: {pin}")
    actions = tracker.snapshot()
    if PROGRESS_MAX_ACTIONS > 0:
        actions = actions[-PROGRESS_MAX_ACTIONS:]
    body_lines = [format_action_line(action) for action in actions]
    if not body_lines:
        return "\n".join(lines)
    return "\n\n".join(["\n".join(lines), "\n".join(body_lines)])


# Progress parsing
def _short_tool_name(server: str | None, tool: str | None) -> str:
    name = ".".join(part for part in (server, tool) if part)
    return name or "tool"


def _normalize_change_list(changes: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for change in changes:
        path: str | None = None
        kind: str | None = None
        if isinstance(change, dict):
            path = change.get("path")
            kind = change.get("kind")
        else:
            path = getattr(change, "path", None)
            kind = getattr(change, "kind", None)
        if not isinstance(path, str) or not path:
            continue
        entry = {"path": path}
        if isinstance(kind, str) and kind:
            entry["kind"] = kind
        normalized.append(entry)
    return normalized


def _phase_from_event(event_type: str) -> str | None:
    if event_type == "item.started":
        return "started"
    if event_type == "item.updated":
        return "updated"
    if event_type == "item.completed":
        return "completed"
    return None


def _parse_progress_update(data: dict[str, Any]) -> ProgressUpdate | None:
    event_type = data.get("type")
    if not isinstance(event_type, str):
        return None
    phase = _phase_from_event(event_type)
    if phase is None:
        return None
    item = data.get("item") or {}
    if not isinstance(item, dict):
        return None
    item_type = item.get("type")
    action_id = item.get("id")
    if not isinstance(action_id, str) or not action_id:
        return None

    if item_type == "command_execution":
        title = str(item.get("command") or "")
        ok = None
        if phase == "completed":
            status = item.get("status")
            ok = status == "completed"
            exit_code = item.get("exit_code")
            if isinstance(exit_code, int):
                ok = ok and exit_code == 0
        return ProgressUpdate(
            action_id=action_id,
            kind="command",
            title=title,
            phase=phase,
            ok=ok,
        )

    if item_type == "mcp_tool_call":
        title = _short_tool_name(item.get("server"), item.get("tool"))
        ok = None
        if phase == "completed":
            status = item.get("status")
            error = item.get("error")
            ok = status == "completed" and error is None
        return ProgressUpdate(
            action_id=action_id,
            kind="tool",
            title=title,
            phase=phase,
            ok=ok,
        )

    if item_type == "file_change":
        changes = item.get("changes")
        if not isinstance(changes, list):
            changes = []
        normalized = _normalize_change_list(changes)
        ok = None
        if phase == "completed":
            ok = item.get("status") == "completed"
        return ProgressUpdate(
            action_id=action_id,
            kind="file_change",
            title="files",
            phase=phase,
            ok=ok,
            detail={"changes": normalized},
        )

    return None


# Codex runner
def build_codex_cmd(
    resume_token: str | None, resume_last: bool, workdir: str
) -> list[str]:
    base = [
        CODEX_CMD,
        "exec",
        "--json",
        "--color",
        "never",
        "--skip-git-repo-check",
        "-C",
        workdir,
    ]
    if resume_token:
        base += ["resume", resume_token, "-"]
    elif resume_last:
        base += ["resume", "--last", "-"]
    else:
        base += ["-"]
    return base


def _is_reconnect_notice(message: str) -> bool:
    return message.strip().lower().startswith("reconnecting")


async def run_codex(
    prompt: str,
    resume_token: str | None,
    resume_last: bool,
    workdir: str,
    on_progress: Callable[[ProgressUpdate], Awaitable[None]] | None = None,
    ssh_host: str | None = None,
) -> CodexRunResult:
    base_cmd = build_codex_cmd(resume_token, resume_last, workdir)
    if ssh_host:
        remote_cmd = shlex.join(base_cmd)
        cmd = ["ssh", "-T", ssh_host, "--", "bash", "-lc", remote_cmd]
    else:
        cmd = base_cmd
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return CodexRunResult(
            message=None,
            resume=None,
            error=f"Codex command not found: {CODEX_CMD}",
            exit_code=None,
        )

    if proc.stdin is None or proc.stdout is None:
        return CodexRunResult(
            message=None,
            resume=None,
            error="codex exec failed to open subprocess pipes",
            exit_code=None,
        )

    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    agent_message: str | None = None
    resume_token: str | None = None
    error_message: str | None = None

    stderr_task = None
    if proc.stderr is not None:
        stderr_task = asyncio.create_task(proc.stderr.read())

    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if on_progress is not None:
            update = _parse_progress_update(data)
            if update is not None:
                await on_progress(update)
        event_type = data.get("type")
        if event_type == "thread.started":
            thread_id = data.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                resume_token = thread_id
            continue
        if event_type == "turn.failed":
            error = (data.get("error") or {}).get("message")
            if isinstance(error, str) and error:
                error_message = error
            continue
        if event_type == "error":
            message = data.get("message")
            if isinstance(message, str) and message and not _is_reconnect_notice(message):
                error_message = message
            continue
        if event_type != "item.completed":
            continue
        item = data.get("item") or {}
        if item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                agent_message = text

    exit_code = await proc.wait()

    stderr_output = ""
    if stderr_task is not None:
        try:
            stderr_bytes = await stderr_task
        except Exception:
            stderr_bytes = b""
        if stderr_bytes:
            stderr_output = stderr_bytes.decode("utf-8", errors="replace").strip()

    if agent_message is None and error_message is None and exit_code != 0:
        error_message = f"codex exec failed (rc={exit_code})"
        if stderr_output:
            error_message = f"{error_message}\n{stderr_output}"

    return CodexRunResult(
        message=agent_message,
        resume=resume_token,
        error=error_message,
        exit_code=exit_code,
    )


def build_prompt(user_text: str, workdir: str) -> str:
    return f"{SYSTEM_HINT}\n\nCurrent working directory: {workdir}\n\nUser: {user_text}\n"


# Resume state persistence
def load_resume_map() -> dict[str, str]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str) and value:
            cleaned[key] = value
    return cleaned


def save_resume_map(resume_map: dict[str, str]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp_path = f"{STATE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(resume_map, handle)
    os.replace(tmp_path, STATE_PATH)


def remember_resume_workdir(state: BotState, token: str) -> None:
    key = f"{state.active_machine}:{token}"
    state.resume_map[key] = state.workdir
    save_resume_map(state.resume_map)


# Project aliases
def load_projects() -> dict[str, str]:
    try:
        with open(PROJECTS_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str) and value:
            cleaned[key] = value
    return cleaned


def save_projects(projects: dict[str, str]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp_path = f"{PROJECTS_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(projects, handle)
    os.replace(tmp_path, PROJECTS_PATH)


def project_key(state: BotState, name: str) -> str:
    return f"{state.active_machine}:{name}"


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_machines() -> dict[str, dict[str, str]]:
    try:
        with open(MACHINES_PATH, "rb") as handle:
            payload = tomllib.load(handle)
    except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("machines")
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, dict[str, str]] = {}
    for name, cfg in raw.items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue
        host = cfg.get("host")
        if not isinstance(host, str) or not host:
            continue
        workdir = cfg.get("workdir")
        entry = {"host": host}
        if isinstance(workdir, str) and workdir:
            entry["workdir"] = workdir
        cleaned[name] = entry
    return cleaned


def save_machines(machines: dict[str, dict[str, str]]) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    lines: list[str] = []
    for name in sorted(machines.keys()):
        cfg = machines[name]
        host = cfg.get("host")
        if not isinstance(host, str) or not host:
            continue
        lines.append(f"[machines.{name}]")
        lines.append(f"host = {_toml_string(host)}")
        workdir = cfg.get("workdir")
        if isinstance(workdir, str) and workdir:
            lines.append(f"workdir = {_toml_string(workdir)}")
        lines.append("")
    tmp_path = f"{MACHINES_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")
    os.replace(tmp_path, MACHINES_PATH)


# Telegram I/O
async def send_chunked(
    application,
    text: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
) -> None:
    if not text:
        return
    for i in range(0, len(text), MAX_TELEGRAM_CHUNK):
        await application.bot.send_message(
            chat_id=chat_id,
            text=text[i : i + MAX_TELEGRAM_CHUNK],
            reply_to_message_id=reply_to_message_id,
        )


def resolve_workdir(target: str, base: str) -> str | None:
    target_path = os.path.expanduser(target)
    if not os.path.isabs(target_path):
        target_path = os.path.abspath(os.path.join(base, target_path))
    return target_path


def resolve_remote_workdir(target: str, base: str) -> str:
    if target.startswith("/") or target.startswith("~"):
        return target
    if base.endswith("/"):
        return base + target
    return posixpath.join(base, target)


def resolve_workdir_for_machine(target: str, base: str, *, remote: bool) -> str:
    if remote:
        return resolve_remote_workdir(target, base)
    return resolve_workdir(target, base) or target


def set_workdir(state: BotState, path: str) -> None:
    state.workdir = path
    state.machine_workdir[state.active_machine] = path


def get_machine_workdir(state: BotState, name: str) -> str | None:
    if name in state.machine_workdir:
        return state.machine_workdir.get(name)
    cfg = state.machines.get(name)
    if isinstance(cfg, dict):
        workdir = cfg.get("workdir")
        if isinstance(workdir, str) and workdir:
            return workdir
    return None


def get_machine_host(state: BotState, name: str) -> str | None:
    if name == "local":
        return None
    cfg = state.machines.get(name)
    if not isinstance(cfg, dict):
        return None
    host = cfg.get("host")
    if isinstance(host, str) and host:
        return host
    return None

async def send_markdown_message(
    application,
    text: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
) -> None:
    if not text:
        return
    body = text.strip()
    if len(body) <= MAX_TELEGRAM_CHUNK:
        try:
            await application.bot.send_message(
                chat_id=chat_id,
                text=body,
                parse_mode=ParseMode.MARKDOWN,
                reply_to_message_id=reply_to_message_id,
            )
            return
        except BadRequest:
            await application.bot.send_message(
                chat_id=chat_id,
                text=body,
                reply_to_message_id=reply_to_message_id,
            )
            return
    await send_chunked(
        application,
        body,
        chat_id=chat_id,
        reply_to_message_id=reply_to_message_id,
    )


async def send_progress_message(
    application, text: str, *, chat_id: int, reply_to_message_id: int
) -> int | None:
    if not text:
        return None
    try:
        message = await application.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            reply_to_message_id=reply_to_message_id,
            disable_notification=True,
        )
    except BadRequest:
        message = await application.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_to_message_id=reply_to_message_id,
            disable_notification=True,
        )
    return getattr(message, "message_id", None)


async def edit_progress_message(
    application, message_id: int, *, chat_id: int, text: str
) -> None:
    try:
        await application.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )
    except BadRequest:
        try:
            await application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
            )
        except BadRequest:
            return


async def delete_message(application, message_id: int, *, chat_id: int) -> None:
    try:
        await application.bot.delete_message(
            chat_id=chat_id,
            message_id=message_id,
        )
    except BadRequest:
        return
    except Exception:
        return


async def send_codex_message(
    application,
    text: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
) -> None:
    if text:
        await send_markdown_message(
            application,
            text.strip(),
            chat_id=chat_id,
            reply_to_message_id=reply_to_message_id,
        )


# Voice transcription
def _normalize_voice_filename(file_path: str | None) -> str:
    name = os.path.basename(file_path or "")
    if not name:
        return "voice.ogg"
    if name.endswith(".oga"):
        return f"{name[:-4]}.ogg"
    return name


def _build_multipart_form(
    *,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    parts.append(data)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def _transcribe_audio_sync(
    *,
    audio_bytes: bytes,
    filename: str,
    mime_type: str,
    model: str,
    api_key: str,
) -> str:
    body, boundary = _build_multipart_form(
        fields={"model": model},
        file_field="file",
        filename=filename,
        content_type=mime_type,
        data=audio_bytes,
    )
    req = urllib.request.Request(
        OPENAI_TRANSCRIPTION_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("transcription returned empty text")
    return text.strip()


async def transcribe_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    if update.message is None or update.message.voice is None:
        return None
    if not VOICE_TRANSCRIPTION:
        await send_codex_message(
            context.application,
            "Voice transcription is disabled.",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return None
    if not OPENAI_API_KEY:
        await send_codex_message(
            context.application,
            "Voice transcription requires OPENAI_API_KEY.",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return None
    voice = update.message.voice
    if voice.file_size is not None and voice.file_size > OPENAI_AUDIO_MAX_BYTES:
        await send_codex_message(
            context.application,
            "Voice message is too large to transcribe.",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return None
    try:
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()
    except Exception:
        await send_codex_message(
            context.application,
            "Failed to download voice message.",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return None
    filename = _normalize_voice_filename(getattr(file, "file_path", None))
    mime_type = voice.mime_type or "audio/ogg"
    try:
        return await asyncio.to_thread(
            _transcribe_audio_sync,
            audio_bytes=bytes(audio_bytes),
            filename=filename,
            mime_type=mime_type,
            model=OPENAI_TRANSCRIPTION_MODEL,
            api_key=OPENAI_API_KEY,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
        await send_codex_message(
            context.application,
            f"Transcription failed: {exc}",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return None


# Resume helpers
def is_authorized(update: Update) -> bool:
    if update.effective_chat is None:
        return False
    return str(update.effective_chat.id) == str(TELEGRAM_CHAT_ID)


def extract_resume_token(text: str | None) -> str | None:
    if not text:
        return None
    found: str | None = None
    for match in RESUME_RE.finditer(text):
        token = match.group("token")
        if token:
            found = token
    return found


def strip_resume_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if RESUME_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def format_resume_line(token: str) -> str:
    return f"`codex resume {token}`"

def append_resume_line(text: str, token: str | None) -> str:
    if not token:
        return text
    if not text:
        return format_resume_line(token)
    if RESUME_RE.search(text):
        return text
    return f"{text.rstrip()}\n\n{format_resume_line(token)}"

# Handlers
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    await update.message.reply_text("Connected to Codex bridge.")


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state: BotState = context.application.bot_data["state"]
    state.has_session = False
    state.resume_token = None
    await update.message.reply_text("Session reset. Next message starts a new Codex session.")


async def pwd_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state: BotState = context.application.bot_data["state"]
    await update.message.reply_text(state.workdir)


async def cd_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state: BotState = context.application.bot_data["state"]
    target = " ".join(context.args).strip()
    if not target:
        await update.message.reply_text("Usage: /cd <path>")
        return
    remote = state.active_machine != "local"
    target_path = resolve_workdir_for_machine(target, state.workdir, remote=remote)
    if remote:
        set_workdir(state, target_path)
        if state.resume_token:
            remember_resume_workdir(state, state.resume_token)
        await update.message.reply_text(state.workdir)
        return
    if target_path and os.path.isdir(target_path):
        set_workdir(state, target_path)
        if state.resume_token:
            remember_resume_workdir(state, state.resume_token)
        await update.message.reply_text(state.workdir)
        return
    await update.message.reply_text(f"No such directory: {target_path or target}")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state: BotState = context.application.bot_data["state"]
    lines = [
        "Status:",
        f"- has_session: {state.has_session}",
        f"- machine: {state.active_machine}",
        f"- workdir: {state.workdir}",
        f"- resume: {state.resume_token or 'none'}",
    ]
    host = get_machine_host(state, state.active_machine)
    if host:
        lines.append(f"- host: {host}")
    if state.resume_token:
        lines.append(f"- resume_line: {format_resume_line(state.resume_token)}")
    if state.pin:
        lines.append(f"- pin: {state.pin}")
    await update.message.reply_text("\n".join(lines))


async def proj_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return
    state: BotState = context.application.bot_data["state"]
    args = context.args
    if not args:
        prefix = f"{state.active_machine}:"
        items = [
            f"{key[len(prefix):]} -> {path}"
            for key, path in sorted(state.project_map.items())
            if key.startswith(prefix)
        ]
        if not items:
            await update.message.reply_text("projects: (empty)")
            return
        await update.message.reply_text("projects:\n" + "\n".join(items))
        return
    if args[0] == "rm":
        if len(args) < 2:
            await update.message.reply_text("Usage: /proj rm <name>")
            return
        name = args[1]
        key = project_key(state, name)
        if key in state.project_map:
            state.project_map.pop(key, None)
            save_projects(state.project_map)
            await update.message.reply_text(f"removed: {name}")
            return
        await update.message.reply_text(f"no such project: {name}")
        return

    name = args[0]
    if len(args) == 1:
        key = project_key(state, name)
        path = state.project_map.get(key)
        if not path:
            await update.message.reply_text(f"no such project: {name}")
            return
        remote = state.active_machine != "local"
        if not remote and not os.path.isdir(path):
            await update.message.reply_text(f"missing directory: {path}")
            return
        set_workdir(state, path)
        if state.resume_token:
            remember_resume_workdir(state, state.resume_token)
        await update.message.reply_text(state.workdir)
        return

    raw_path = " ".join(args[1:]).strip()
    if not raw_path:
        await update.message.reply_text("Usage: /proj <name> <path>")
        return
    remote = state.active_machine != "local"
    path = resolve_workdir_for_machine(raw_path, state.workdir, remote=remote)
    if not remote and (not path or not os.path.isdir(path)):
        await update.message.reply_text(f"No such directory: {path or raw_path}")
        return
    state.project_map[project_key(state, name)] = path
    save_projects(state.project_map)
    set_workdir(state, path)
    if state.resume_token:
        remember_resume_workdir(state, state.resume_token)
    await update.message.reply_text(state.workdir)


async def machine_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state: BotState = context.application.bot_data["state"]
    args = context.args
    if not args:
        lines: list[str] = []
        active = state.active_machine
        local_prefix = "*" if active == "local" else "-"
        lines.append(
            f"{local_prefix} local -> {state.machine_workdir.get('local', DEFAULT_WORKDIR)}"
        )
        for name in sorted(state.machines.keys()):
            cfg = state.machines[name]
            host = cfg.get("host", "")
            workdir = get_machine_workdir(state, name) or ""
            prefix = "*" if name == active else "-"
            lines.append(f"{prefix} {name} -> {host} {workdir}")
        await update.message.reply_text("machines:\n" + "\n".join(lines))
        return
    if args[0] == "current":
        await update.message.reply_text(f"machine: {state.active_machine}")
        return

    name = args[0]
    if name != "local" and name not in state.machines:
        await update.message.reply_text(f"no such machine: {name}")
        return
    workdir = get_machine_workdir(state, name)
    if not workdir:
        await update.message.reply_text(f"missing workdir for machine: {name}")
        return
    state.active_machine = name
    set_workdir(state, workdir)
    if state.resume_token:
        remember_resume_workdir(state, state.resume_token)
    await update.message.reply_text(f"machine: {name}\n{state.workdir}")


async def pin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state: BotState = context.application.bot_data["state"]
    text = " ".join(context.args).strip()
    if not text:
        if state.pin:
            await update.message.reply_text(f"pin: {state.pin}")
        else:
            await update.message.reply_text("pin: (empty)")
        return
    state.pin = shorten(text, 120)
    await update.message.reply_text(f"pin set: {state.pin}")


async def unpin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    state: BotState = context.application.bot_data["state"]
    state.pin = None
    await update.message.reply_text("pin cleared.")


async def run_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return
    state: BotState = context.application.bot_data["state"]
    cmd = " ".join(context.args).strip()
    if not cmd:
        await update.message.reply_text("Usage: /run <command>")
        return
    if state.run_lock.locked():
        await send_codex_message(
            context.application,
            "Codex is busy. Please wait for the current run to finish.",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return
    try:
        ssh_host = get_machine_host(state, state.active_machine)
        if ssh_host:
            remote_cmd = f"cd {shlex.quote(state.workdir)} && {cmd}"
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-T",
                ssh_host,
                "--",
                "bash",
                "-lc",
                remote_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=state.workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
    except Exception as exc:
        await update.message.reply_text(f"Failed to start: {exc}")
        return
    stdout, stderr = await proc.communicate()
    out = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
    err = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
    parts: list[str] = []
    parts.append(f"$ {cmd}")
    if out:
        parts.append(out)
    if err:
        parts.append(err)
    parts.append(f"(exit {proc.returncode})")
    message = "\n".join(parts)
    await send_codex_message(
        context.application,
        message,
        chat_id=int(update.effective_chat.id),
        reply_to_message_id=update.message.message_id,
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if update.message is None:
        return
    if update.message.text is None and update.message.voice is None:
        return
    if update.effective_chat is None:
        return

    state: BotState = context.application.bot_data["state"]
    if state.run_lock.locked():
        await send_codex_message(
            context.application,
            "Codex is busy. Please wait for the current run to finish.",
            chat_id=int(update.effective_chat.id),
            reply_to_message_id=update.message.message_id,
        )
        return

    async with state.run_lock:
        await handle_run(update, context)


async def handle_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    if update.message.text is None and update.message.voice is None:
        return
    if update.effective_chat is None:
        return
    state: BotState = context.application.bot_data["state"]
    chat_id = int(update.effective_chat.id)
    raw_text = update.message.text
    if raw_text is None or not raw_text.strip():
        if update.message.voice is not None:
            transcribed = await transcribe_voice(update, context)
            if not transcribed:
                return
            text = transcribed
        else:
            return
    else:
        text = raw_text.strip()

    reply_text = None
    if update.message.reply_to_message is not None:
        reply_text = update.message.reply_to_message.text

    explicit_resume = extract_resume_token(text) or extract_resume_token(reply_text)
    stripped_text = strip_resume_lines(text).strip()
    if not stripped_text:
        stripped_text = "continue"
    resume_token = explicit_resume or (state.resume_token if state.has_session else None)
    if explicit_resume:
        key = f"{state.active_machine}:{explicit_resume}"
        mapped = state.resume_map.get(key)
        remote = state.active_machine != "local"
        if isinstance(mapped, str) and mapped and (remote or os.path.isdir(mapped)):
            set_workdir(state, mapped)
    resume_last = state.has_session and resume_token is None
    started_at = time.monotonic()
    progress_tracker = ProgressTracker()
    progress_label = "working"
    last_rendered: str | None = None
    last_edit_at = 0.0
    progress_queue: asyncio.Queue[ProgressUpdate] | None = None
    progress_task: asyncio.Task[None] | None = None
    progress_message_id = None
    host_label = get_machine_host(state, state.active_machine)
    try:
        progress_message_id = await send_progress_message(
            context.application,
            render_progress(
                progress_tracker,
                elapsed_s=0.0,
                label=progress_label,
                pin=state.pin,
                host=host_label,
            ),
            chat_id=chat_id,
            reply_to_message_id=update.message.message_id,
        )
    except Exception:
        progress_message_id = None

    async def maybe_edit_progress(force: bool = False) -> None:
        nonlocal last_rendered, last_edit_at
        if progress_message_id is None:
            return
        now = time.monotonic()
        if not force and now - last_edit_at < PROGRESS_EDIT_MIN_INTERVAL_S:
            return
        rendered = render_progress(
            progress_tracker,
            elapsed_s=now - started_at,
            label=progress_label,
            pin=state.pin,
            host=host_label,
        )
        if rendered == last_rendered:
            return
        try:
            await edit_progress_message(
                context.application,
                progress_message_id,
                chat_id=chat_id,
                text=rendered,
            )
        except Exception:
            return
        last_rendered = rendered
        last_edit_at = now

    async def on_progress(update: ProgressUpdate) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait(update)
        except asyncio.QueueFull:
            return

    async def progress_loop() -> None:
        nonlocal progress_label
        if progress_queue is None:
            return
        try:
            while True:
                try:
                    update = await asyncio.wait_for(
                        progress_queue.get(), timeout=PROGRESS_TICK_S
                    )
                except asyncio.TimeoutError:
                    update = None
                if update is not None:
                    if progress_tracker.note(update):
                        progress_label = "working"
                await maybe_edit_progress()
        except asyncio.CancelledError:
            return

    prompt = build_prompt(stripped_text, state.workdir)
    if progress_message_id is not None:
        progress_queue = asyncio.Queue(maxsize=200)
        progress_task = asyncio.create_task(progress_loop())
    ssh_host = get_machine_host(state, state.active_machine)
    result = await run_codex(
        prompt,
        resume_token=resume_token,
        resume_last=resume_last,
        workdir=state.workdir,
        on_progress=on_progress if progress_message_id is not None else None,
        ssh_host=ssh_host,
    )
    if progress_task is not None:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    if progress_message_id is not None:
        await delete_message(context.application, progress_message_id, chat_id=chat_id)
    state.has_session = True
    if result.resume:
        state.resume_token = result.resume
        remember_resume_workdir(state, result.resume)
    if result.message:
        message = result.message
        if result.error:
            message = f"{message}\n\n{result.error}"
        message = append_resume_line(message, state.resume_token)
        await send_codex_message(
            context.application,
            message or "",
            chat_id=chat_id,
            reply_to_message_id=update.message.message_id,
        )
        return
    if result.error:
        message = append_resume_line(result.error, state.resume_token)
        await send_codex_message(
            context.application,
            message,
            chat_id=chat_id,
            reply_to_message_id=update.message.message_id,
        )


# Entrypoint
def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    state = BotState()
    state.resume_map = load_resume_map()
    state.project_map = load_projects()
    state.machines = load_machines()
    state.machine_workdir["local"] = DEFAULT_WORKDIR
    for name, cfg in state.machines.items():
        workdir = cfg.get("workdir")
        if isinstance(workdir, str) and workdir:
            state.machine_workdir[name] = workdir
    application.bot_data["state"] = state
    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("reset", reset_cmd))
    application.add_handler(CommandHandler("pwd", pwd_cmd))
    application.add_handler(CommandHandler("cd", cd_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("proj", proj_cmd))
    application.add_handler(CommandHandler("pin", pin_cmd))
    application.add_handler(CommandHandler("unpin", unpin_cmd))
    application.add_handler(CommandHandler("run", run_cmd))
    application.add_handler(
        MessageHandler((filters.TEXT | filters.VOICE) & ~filters.COMMAND, on_message)
    )
    application.run_polling()


if __name__ == "__main__":
    main()
