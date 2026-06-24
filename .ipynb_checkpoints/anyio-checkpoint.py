from __future__ import annotations

import re as _re
import socket as _socket
import ssl
import typing

import anyio

from .._exceptions import (
    ConnectError,
    ConnectTimeout,
    ReadError,
    ReadTimeout,
    WriteError,
    WriteTimeout,
    map_exceptions,
)
from .._utils import is_socket_readable
from .base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream


class AnyIOStream(AsyncNetworkStream):
    def __init__(self, stream: anyio.abc.ByteStream) -> None:
        self._stream = stream

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        exc_map = {
            TimeoutError: ReadTimeout,
            anyio.BrokenResourceError: ReadError,
            anyio.ClosedResourceError: ReadError,
            anyio.EndOfStream: ReadError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                try:
                    return await self._stream.receive(max_bytes=max_bytes)
                except anyio.EndOfStream:  # pragma: nocover
                    return b""

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if not buffer:
            return

        exc_map = {
            TimeoutError: WriteTimeout,
            anyio.BrokenResourceError: WriteError,
            anyio.ClosedResourceError: WriteError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                await self._stream.send(item=buffer)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        exc_map = {
            TimeoutError: ConnectTimeout,
            anyio.BrokenResourceError: ConnectError,
            anyio.EndOfStream: ConnectError,
            ssl.SSLError: ConnectError,
        }
        with map_exceptions(exc_map):
            try:
                with anyio.fail_after(timeout):
                    ssl_stream = await anyio.streams.tls.TLSStream.wrap(
                        self._stream,
                        ssl_context=ssl_context,
                        hostname=server_hostname,
                        standard_compatible=False,
                        server_side=False,
                    )
            except Exception as exc:  # pragma: nocover
                await self.aclose()
                raise exc
        return AnyIOStream(ssl_stream)

    def get_extra_info(self, info: str) -> typing.Any:
        if info == "ssl_object":
            return self._stream.extra(anyio.streams.tls.TLSAttribute.ssl_object, None)
        if info == "client_addr":
            return self._stream.extra(anyio.abc.SocketAttribute.local_address, None)
        if info == "server_addr":
            return self._stream.extra(anyio.abc.SocketAttribute.remote_address, None)
        if info == "socket":
            return self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
        if info == "is_readable":
            sock = self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
            return is_socket_readable(sock)
        return None


def _is_raw_ipv4(host: str) -> bool:
    return bool(_re.match(r"^\d{1,3}(\.\d{1,3}){3}$", str(host)))


async def _connect_tcp_via_socket(
    host: str,
    port: int,
    timeout: float | None,
    socket_options: typing.Iterable[SOCKET_OPTION],
) -> anyio.abc.ByteStream:
    """
    Fallback for environments where uvloop's DNS resolver can't reach
    Docker's 127.0.0.11 nameserver from subprocesses. Uses synchronous
    socket.create_connection (which bypasses the async resolver) and
    wraps the result as an anyio SocketStream.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    # Run the blocking connect in a thread so we don't block the event loop
    sock = await loop.run_in_executor(
        None,
        lambda: _socket.create_connection((host, port), timeout=timeout)
    )
    sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
    for option in socket_options:
        sock.setsockopt(*option)
    return anyio.from_socket(sock)


class AnyIOBackend(AsyncNetworkBackend):
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:  # pragma: nocover
        if socket_options is None:
            socket_options = []
        exc_map = {
            TimeoutError: ConnectTimeout,
            OSError: ConnectError,
            anyio.BrokenResourceError: ConnectError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                # uvloop's async DNS resolver can't reach Docker's 127.0.0.11
                # nameserver from subprocesses, even for raw IPs. For raw IPv4
                # addresses, bypass the async resolver entirely and use the
                # synchronous socket.create_connection which works fine.
                if _is_raw_ipv4(host):
                    stream = await _connect_tcp_via_socket(
                        host, port, timeout, socket_options
                    )
                else:
                    stream = await anyio.connect_tcp(
                        remote_host=host,
                        remote_port=port,
                        local_host=local_address,
                    )
                    for option in socket_options:
                        stream._raw_socket.setsockopt(*option)  # type: ignore[attr-defined]
        return AnyIOStream(stream)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:  # pragma: nocover
        if socket_options is None:
            socket_options = []
        exc_map = {
            TimeoutError: ConnectTimeout,
            OSError: ConnectError,
            anyio.BrokenResourceError: ConnectError,
        }
        with map_exceptions(exc_map):
            with anyio.fail_after(timeout):
                stream: anyio.abc.ByteStream = await anyio.connect_unix(path)
                for option in socket_options:
                    stream._raw_socket.setsockopt(*option)  # type: ignore[attr-defined] # pragma: no cover
        return AnyIOStream(stream)

    async def sleep(self, seconds: float) -> None:
        await anyio.sleep(seconds)  # pragma: nocover