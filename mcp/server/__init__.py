# Minimal implementation of the main MCP Server class.
import asyncio
from functools import wraps
from .stdio import read_message, write_message


class Tool:
    def __init__(self, name, description, inputSchema):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


class Server:
    def __init__(self, name):
        self.name = name
        self._list_tools_func = None
        self._call_tool_func = None

    def list_tools(self):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._list_tools_func = wrapper
            return wrapper

        return decorator

    def call_tool(self):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._call_tool_func = wrapper
            return wrapper

        return decorator

    def create_initialization_options(self):
        # This would normally be more dynamic
        return {"name": self.name}

    async def run(self, reader, writer, initialization_options):
        # Handle initialization
        init_req = await read_message(reader)
        if init_req and init_req.get("method") == "initialize":
            await write_message(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": init_req["id"],
                    "result": {"serverInfo": initialization_options},
                },
            )

        # Main message loop
        while True:
            request = await read_message(reader)
            if request is None:
                break

            method = request.get("method")
            response = {"jsonrpc": "2.0", "id": request.get("id")}

            try:
                if method == "tools/list":
                    tools = await self._list_tools_func()
                    response["result"] = [t.to_dict() for t in tools]
                elif method == "tools/call":
                    params = request.get("params", {})
                    result = await self._call_tool_func(
                        name=params.get("name"), arguments=params.get("arguments")
                    )
                    response["result"] = result
                else:
                    response["error"] = {"code": -32601, "message": "Method not found"}
            except Exception as e:
                response["error"] = {"code": -32603, "message": f"Internal error: {e}"}

            await write_message(writer, response)
