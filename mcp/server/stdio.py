# Minimal implementation of an MCP stdio server based on the prompt's example.
import asyncio
import json
import sys


async def read_message(stream):
    line = await stream.readline()
    if not line:
        return None
    line = line.decode("utf-8").strip()
    if line.startswith("Content-Length:"):
        length = int(line.split(":")[1].strip())
        await stream.readline()  # Consume the blank line
        body = await stream.readexactly(length)
        return json.loads(body.decode("utf-8"))
    return None


async def write_message(stream, message):
    body = json.dumps(message).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream.write(header + body)
    await stream.drain()


class StdioServer:
    def __init__(self):
        self.reader = None
        self.writer = None

    async def __aenter__(self):
        loop = asyncio.get_event_loop()
        self.reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self.reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        self.writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, None, loop
        )
        return self.reader, self.writer

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()


def stdio_server():
    return StdioServer()
