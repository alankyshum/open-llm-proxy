# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately through [GitHub Security Advisories](https://github.com/alankyshum/open-llm-proxy/security/advisories/new). Do not disclose sensitive details in a public issue.

## Supported versions

Only the latest released version on the default branch is supported with security fixes.

## Deployment warning

This proxy brokers OAuth tokens and API keys, stores credentials under
`~/.config/open-llm-proxy/`, and spools attachment bytes to disk. It must not be
exposed to untrusted networks. Bind it to a trusted interface and protect any
administrative or upstream credentials appropriately.
