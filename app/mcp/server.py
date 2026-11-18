"""
MCP server: exposes the read-only database tools over stdio.

Agents never import this file. They start it as a separate process through
app/mcp/client.py, so the database stays behind the MCP boundary.

Manual run (for debugging only): python -m app.mcp.server
"""

from mcp.server.fastmcp import FastMCP

from app.mcp.tools import TOOLS

mcp = FastMCP("mediassist-db")

for _tool in TOOLS:  # register every tool; the docstring becomes its description
    mcp.tool()(_tool)

if __name__ == "__main__":
    mcp.run(transport="stdio")
