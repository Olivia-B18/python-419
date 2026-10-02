"""
Convert an exported Claude Code transcript (.jsonl) into readable Markdown.

Keeps your prompts and Claude's text replies. Skips tool calls, tool
results, thinking blocks, and app bookkeeping lines.

Usage (from the homework1 directory):
    unzip -p ~/Downloads/session-export-1790969963993.zip transcript.jsonl > transcript.jsonl
    python transcript_to_md.py transcript.jsonl claude_convo.md

By Claude Code (Opus 5.5) on 10/2/2026
"""
import json
import re
import sys
from pathlib import Path

# Context the app attaches to prompts; not something you typed
SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

# Pasted text is wrapped in tags carrying an id; keep the text, drop the tags
PASTED_TAG = re.compile(r"</?pasted_content[^>]*>")


def clean_prompt(text: str) -> str:
    """Remove app-added context and tags from a prompt."""
    text = SYSTEM_REMINDER.sub("", text)
    text = PASTED_TAG.sub("", text)
    return text.strip()


def extract_turns(jsonl_path: Path) -> list[tuple[str, str]]:
    """
    Read the transcript and return (speaker, text) pairs in order.

    Prompts are stored as plain strings. Claude's replies are lists of
    parts, and only the "text" parts are what you saw in the chat.
    """
    turns = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") not in ("user", "assistant"):
                continue

            content = record["message"]["content"]

            # Tool results are also stored as "user" records, but as lists
            if record["type"] == "user" and isinstance(content, str):
                text = clean_prompt(content)
                if text:
                    turns.append(("User", text))

            elif record["type"] == "assistant":
                for part in content:
                    if part.get("type") == "text" and part["text"].strip():
                        turns.append(("Claude", part["text"].strip()))
    return turns


def write_markdown(turns: list[tuple[str, str]], md_path: Path):
    """Write turns to Markdown, merging consecutive parts by the same speaker."""
    lines = ["# Conversation with Claude Code", ""]
    previous_speaker = None
    for speaker, text in turns:
        if speaker != previous_speaker:
            lines.extend(["---", "", f"## {speaker}", ""])
            previous_speaker = speaker
        lines.extend([text, ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python transcript_to_md.py <transcript.jsonl> <output.md>")

    turns = extract_turns(Path(sys.argv[1]))
    write_markdown(turns, Path(sys.argv[2]))
    print(f"Wrote {len(turns)} parts to {sys.argv[2]}")
