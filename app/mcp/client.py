"""
MCP client: starts the server process, discovers its tools, calls them.

Usage from an agent:
    from app.mcp.client import call_tool_sync
    data = call_tool_sync("get_lab_results", {"patient_id": "P00001"})
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app import config

SERVER = StdioServerParameters(
    command=sys.executable,  # same Python as the one running this code
    args=["-m", "app.mcp.server"],
    cwd=str(config.PROJECT_ROOT),
)


async def list_tools() -> list[dict]:
    """Ask the server which tools exist (discovered at runtime, not hard-coded)."""
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [{"name": t.name, "description": t.description} for t in result.tools]


async def call_tool(name: str, arguments: dict) -> dict:
    """Call one tool and return its result as a dictionary."""
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            if getattr(result, "structuredContent", None):
                data = result.structuredContent
                return data.get("result", data) if isinstance(data, dict) else data
            text = "".join(getattr(c, "text", "") for c in result.content)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": text or "The tool returned nothing."}


def call_tool_sync(name: str, arguments: dict) -> dict:  # for normal (non-async) code
    return asyncio.run(call_tool(name, arguments))


def list_tools_sync() -> list[dict]:
    return asyncio.run(list_tools())
