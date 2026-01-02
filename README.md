# codie

Telegram bot bridge for the [Codex CLI](https://developers.openai.com/codex/cli). Runs tools locally in a persistent working directory.

## Features
- Single authorized chat controls a stateful Codex session
- Live progress updates (commands/tools/files)
- `/cd`, `/pwd`, `/status`, `/reset`
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

## Commands
- `/start`: confirm connectivity
- `/reset`: reset the Codex session state
- `/pwd`: show current working directory
- `/cd <path>`: change working directory (supports relative paths)
- `/status`: show session state and working directory (includes resume token)
  - Resume token can be used in a message: `codex resume <token>`

## Voice Notes
If enabled, voice notes are transcribed via OpenAI and sent to Codex as text.

## Resume
Resume tokens restore the last known working directory for that session. The mapping is stored at `~/.codie/state.json`.

## Security Notes
This bot can run shell commands and edit files. Keep the bot token private and restrict `TELEGRAM_CHAT_ID`.

## License
MIT
