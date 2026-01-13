# codie

Telegram bot bridge for the [Codex CLI](https://developers.openai.com/codex/cli). Runs tools locally in a persistent working directory.

## Features
- Single authorized chat controls a stateful Codex session
- Live progress updates (commands/tools/files)
- `/cd`, `/pwd`, `/status`, `/reset`, `/pin`, `/unpin`, `/run`, `/proj`, `/machine`
- Optional voice note transcription
- Resume sessions with `codex resume <token>` and restore last workdir

## Quickstart
Prereqs: Codex CLI on PATH.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py
```

## Configuration
Required:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID` (single chat allowed)

Optional:
- `CODEX_CMD` (default: `codex`)
- `CODEX_WORKDIR` (default: current directory)
- `VOICE_TRANSCRIPTION` (`true` to enable)
- `OPENAI_API_KEY` (required for voice transcription)
- `OPENAI_TRANSCRIPTION_MODEL` (default: `gpt-4o-mini-transcribe`)

Codex CLI config (optional): `~/.codex/config.toml`
```toml
sandbox_mode = "workspace-write"
approval_policy = "on-request"

[sandbox_workspace_write]
network_access = true
```

## Commands
- `/start`: confirm connectivity
- `/help`: show the command list
- `/reset`: reset the Codex session state
- `/pwd`: show current working directory
- `/cd <path>`: change working directory (supports relative paths)
- `/status`: show session state and working directory (includes resume token)
  - Resume token can be used in a message: `codex resume <token>`
- `/machine`: list machines
- `/machine <name>`: switch active machine (default is `local`)
- `/machine current`: show active machine
- `/proj`: list project aliases
- `/proj <name>`: switch to a saved project
- `/proj <name> <path>`: save (and switch to) a project alias
- `/proj rm <name>`: remove a project alias
- `/pin <text>`: set a short context note shown during runs
- `/pin`: show current pin
- `/unpin`: clear the pin
- `/run <command>`: run a shell command in the current working directory

## Voice Notes
If enabled, voice notes are transcribed via OpenAI and sent to Codex as text.

## Resume
Resume tokens restore the last known working directory for that session. The mapping is stored at `~/.codie/state.json`.

## Projects
Project aliases are stored at `~/.codie/projects.json`.

## Machines
Codie can run Codex and `/run` on remote machines via SSH (works well with Tailscale).
Set up aliases in `~/.codie/machines.toml`, then select with `/machine <name>`.
`local` is the default target.

Example:

```toml
[machines.laptop]
host = "laptop"           # can be user@host
workdir = "/home/me/code"

[machines.desktop]
host = "desktop"
workdir = "/home/me/projects"
```

Notes:
- Codex must be installed and on PATH on each machine.
- Remote paths are not validated; make sure they exist.
- Resume tokens are tracked per machine and restore the last known workdir for that machine.

## Security Notes
This bot can run shell commands and edit files. Keep the bot token private and restrict `TELEGRAM_CHAT_ID`.

## License
MIT
