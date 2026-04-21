import asyncio
import sys
from pathlib import Path

# Add the project root to the path to allow importing nedster tools
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server, Tool
from mcp.server.stdio import stdio_server

# Import Nedster's actual tool functions
# Note: Some functions might need slight adaptations to be async or to fit the server model
from tools import run_bash, grep_search, read_file
from code_tools import edit_file, list_files_recursive


# A simple shim for glob_search which is not directly in tools
async def glob_search(pattern: str, root: str = "."):
    # list_files_recursive is a good-enough proxy for a simple glob
    return list_files_recursive(directory=root, pattern=pattern)


# Mock for scaffold_project as it's a more complex workflow
async def scaffold_project(description: str):
    return "Project scaffolding is a complex task and not directly exposed. Please use individual file tools."


server = Server("nedster")


@server.list_tools()
async def list_tools():
    """Lists the tools exposed by the Nedster MCP server."""
    return [
        Tool(
            name="run_bash",
            description="Executes a bash command and returns the output.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The command to execute."}
                },
                "required": ["cmd"],
            },
        ),
        Tool(
            name="glob_search",
            description="Find files matching a glob pattern.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to search for.",
                    },
                    "root": {
                        "type": "string",
                        "description": "The directory to start searching from.",
                    },
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="grep_search",
            description="Search for a regex pattern in files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex pattern."},
                    "path": {
                        "type": "string",
                        "description": "The file or directory to search in.",
                    },
                },
                "required": ["pattern", "path"],
            },
        ),
        Tool(
            name="read_file",
            description="Reads the entire content of a file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file."}
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="edit_file",
            description="Performs an exact string replacement in a file. VERIFIED ON DISK.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file."},
                    "old_str": {
                        "type": "string",
                        "description": "The exact string to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The new string to insert.",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        ),
        Tool(
            name="scaffold_project",
            description="A placeholder for a high-level scaffolding tool.",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A description of the project to scaffold.",
                    }
                },
                "required": ["description"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Dispatches tool calls to the appropriate Nedster function."""
    # Using asyncio.to_thread for synchronous tool functions
    if name == "run_bash":
        return await asyncio.to_thread(run_bash, **arguments)
    elif name == "glob_search":
        return await glob_search(**arguments)
    elif name == "grep_search":
        return await asyncio.to_thread(grep_search, **arguments)
    elif name == "read_file":
        return await asyncio.to_thread(read_file, **arguments)
    elif name == "edit_file":
        # edit_file in code_tools expects 'path', 'old_string', 'new_string'
        mapped_args = {
            "path": arguments["path"],
            "old_string": arguments["old_str"],
            "new_string": arguments["new_str"],
        }
        return await asyncio.to_thread(edit_file, **mapped_args)
    elif name == "scaffold_project":
        return await scaffold_project(**arguments)
    else:
        raise ValueError(f"Tool '{name}' not found.")


async def main():
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
