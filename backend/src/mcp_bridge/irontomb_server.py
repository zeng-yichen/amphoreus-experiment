#!/usr/bin/env python3
"""MCP server exposing Irontomb's retrieval tools.

Launched as a subprocess by the Claude CLI via --mcp-config.
Reads the company keyword from the IRONTOMB_COMPANY env var
(set by the caller before spawning).

Tools exposed:
  - search_past_posts(query, limit)
  - get_recent_posts(limit)
  - get_post_detail(ordinal_post_id)
  - search_linkedin_corpus(query, mode, limit)
  - submit_reaction(...) — terminal tool, returns the prediction as-is

The submit_reaction tool doesn't do anything server-side — it just
echoes back the arguments. The Claude CLI model calls it when it's
ready to submit its prediction, and the caller parses the result
from the CLI output.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on sys.path so imports work when launched
# as a standalone subprocess.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.src.mcp_bridge.server import MCPServer

_COMPANY = os.environ.get("IRONTOMB_COMPANY", "")

server = MCPServer("irontomb-tools")


# ---------------------------------------------------------------------------
# Tool: search_past_posts
# ---------------------------------------------------------------------------
server.register(
    name="search_past_posts",
    description=(
        "Keyword search over this client's scored post history. "
        "Returns posts ranked by relevance to the query, each with "
        "real engagement metrics (reactions, impressions, comments, "
        "reposts), the draft text, published text, and reactor list."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (1-10, default 3)"},
        },
        "required": ["query"],
    },
    handler=lambda args: _dispatch("search_past_posts", args),
)


# ---------------------------------------------------------------------------
# Tool: get_recent_posts
# ---------------------------------------------------------------------------
server.register(
    name="get_recent_posts",
    description=(
        "Get the N most recent scored posts from this client. "
        "Returns the same fields as search_past_posts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "How many posts (1-10, default 3)"},
        },
    },
    handler=lambda args: _dispatch("get_recent_posts", args),
)


# ---------------------------------------------------------------------------
# Tool: get_post_detail
# ---------------------------------------------------------------------------
server.register(
    name="get_post_detail",
    description=(
        "Get the full untruncated text + complete reactor list for "
        "one specific post, identified by ordinal_post_id."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ordinal_post_id": {"type": "string"},
        },
        "required": ["ordinal_post_id"],
    },
    handler=lambda args: _dispatch("get_post_detail", args),
)


# ---------------------------------------------------------------------------
# Tool: search_linkedin_corpus
# ---------------------------------------------------------------------------
server.register(
    name="search_linkedin_corpus",
    description=(
        "Search 200K+ real LinkedIn posts from creators across all "
        "industries. See what performs well LinkedIn-wide for a given "
        "topic or angle. Supports keyword and semantic modes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["keyword", "semantic"],
                "description": "Search mode (default: keyword)",
            },
            "limit": {"type": "integer", "description": "Max results (1-20, default 10)"},
        },
        "required": ["query"],
    },
    handler=lambda args: _dispatch("search_linkedin_corpus", args),
)


# ---------------------------------------------------------------------------
# Tool: submit_reaction (terminal)
# ---------------------------------------------------------------------------
server.register(
    name="submit_reaction",
    description=(
        "Submit your reader reaction. Ends the session.\n"
        "  reaction — under-15-word GESTALT reader-voice reaction "
        "(net effect after reading the whole draft)\n"
        "  anchors  — list of {quote, reaction} for each reader-state-"
        "change moment. At least 1 required."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "reaction": {
                "type": "string",
                "description": (
                    "Your raw, visceral, under-15-word GESTALT reaction "
                    "as a LinkedIn reader — the thing that flashes "
                    "through your head AFTER reading the last line. "
                    "Captures the post's net effect (accumulated "
                    "irritation, lingering felt-ness, the moment it "
                    "died). Not a critique. Not writing-teacher "
                    "vocabulary."
                ),
            },
            "anchors": {
                "type": "array",
                "minItems": 1,
                "description": (
                    "Reader-state-change moments as you read. At least "
                    "one anchor required — surface the specific phrase "
                    "where your felt-state shifted (positive OR "
                    "negative). For drafts with multiple shifts, emit "
                    "one anchor per shift in reading order. For drafts "
                    "that read uniformly without specific moments "
                    "standing out, anchor on the line that best "
                    "represents the overall texture."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {
                            "type": "string",
                            "description": (
                                "3-15 words verbatim from the draft — "
                                "the phrase that triggered the shift."
                            ),
                        },
                        "reaction": {
                            "type": "string",
                            "description": (
                                "Your under-15-word reader-voice "
                                "reaction to that specific moment. "
                                "Same register as the gestalt reaction."
                            ),
                        },
                    },
                    "required": ["quote", "reaction"],
                },
            },
        },
        "required": ["reaction", "anchors"],
    },
    # submit_reaction just echoes back — the caller parses it from the output
    handler=lambda args: json.dumps({"submitted": True, **args}),
)


# ---------------------------------------------------------------------------
# Dispatcher — delegates to Irontomb's existing _dispatch_retrieval_tool
# ---------------------------------------------------------------------------

def _dispatch(tool_name: str, args: dict) -> str:
    from backend.src.agents.irontomb import _dispatch_retrieval_tool
    return _dispatch_retrieval_tool(_COMPANY, tool_name, args)


if __name__ == "__main__":
    if not _COMPANY:
        print("Error: IRONTOMB_COMPANY env var not set", file=sys.stderr)
        sys.exit(1)
    server.run()
