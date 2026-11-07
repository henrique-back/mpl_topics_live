"""AI-written text (topic labels, the weekly newsletter), from Claude through the headless `claude` CLI.

Runs `claude -p` with no tools, the model config.CLAUDE_MODEL and a JSON schema for the answer, so it uses the Claude
Code login of this machine and needs no API key. It runs in an empty temporary folder, so no project instructions
(CLAUDE.md) are loaded, and with its own system prompt instead of Claude Code's.
"""
import json
import subprocess
import tempfile

import config

TIMEOUT = 30 * 60  # seconds


def ask(prompt, schema, system):
    """Claude's answer to `prompt` (sent on stdin), as the dict matching the JSON schema `schema`."""
    command = ['claude', '-p', '--model', config.CLAUDE_MODEL, '--output-format', 'json',
               '--json-schema', json.dumps(schema), '--tools', '', '--no-session-persistence',
               '--system-prompt', system]
    with tempfile.TemporaryDirectory() as folder:
        done = subprocess.run(command, input=prompt, capture_output=True, text=True, cwd=folder, timeout=TIMEOUT)
    try:
        reply = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude failed (exit {done.returncode}): {done.stderr.strip() or done.stdout[:500]}")
    if reply.get('is_error') or reply.get('structured_output') is None:
        raise RuntimeError(f"claude returned no answer: {reply.get('subtype')} {str(reply.get('result'))[:500]}")
    return reply['structured_output']
