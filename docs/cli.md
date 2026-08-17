# CLI reference

The installed command is `open-llm-proxy`. Use `open-llm-proxy help` or `open-llm-proxy help <command>` for the live command help.

| Command | Purpose |
| --- | --- |
| `serve` | Start the OpenAI-compatible proxy |
| `ui --open` | Check the Admin UI and open it in a browser |
| `start`, `stop`, `restart`, `status` | Manage the configured service |
| `setup` | Configure provider rate-limit plans |
| `config --format yaml\|json` | Generate the LiteLLM router configuration |
| `reload` | Hot-reload routing configuration |
| `models [provider]` | List exact model IDs; supports `--search` and `--format json` |
| `auth` | Manage provider credentials and accounts |

Common examples:

```bash
open-llm-proxy serve --port 8765
open-llm-proxy config --format yaml
open-llm-proxy models openrouter --search <text>
open-llm-proxy setup --config <path> --non-interactive
open-llm-proxy reload --config <path>
```

`serve` accepts `--host`, `--port`, `--config`, `--disable-admin-ui`, `--database-url`, `--master-key`, `--ui-username`, and `--ui-password`. Secret flags work for compatibility but are deprecated; use environment variables instead.
