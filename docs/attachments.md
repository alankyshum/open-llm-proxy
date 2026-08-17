# Attachments

The proxy normalizes chat attachment parts before forwarding them. Standard `text` and `image_url` parts are retained; image attachments are converted to `image_url`, and text-like data URIs are decoded inline. Images sent to `/responses`-routed GitHub Copilot models use the `input_image` shape. Set `OPEN_LLM_PROXY_NORMALIZE_ATTACHMENTS=0` (also `false` or `no`) to disable normalization.

Non-renderable attachments such as PDFs, archives, and binary data are written to disk and represented to the agent by an absolute path. Spooling is content-addressed, atomic, best effort, and uses a `0700` directory with `0600` files. The default directory is `~/.local/share/open-llm-proxy/attachments`.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OPEN_LLM_PROXY_SPOOL_ATTACHMENTS` | `1` | Set to `0`, `false`, or `no` to disable file spooling. |
| `OPEN_LLM_PROXY_ATTACHMENT_DIR` | `~/.local/share/open-llm-proxy/attachments` | Directory for spooled files. |
| `OPEN_LLM_PROXY_ATTACHMENT_RETENTION_DAYS` | `7` | Delete older files after writes; `0` disables pruning. |
