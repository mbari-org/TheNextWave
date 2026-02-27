"""
Ethernet bridge reader / decoder.

Provides a small iterator that scans for SBG sync bytes and yields message
headers so callers can delegate payload parsing to `sbgMessageParse`.
"""

from __future__ import annotations

import socket
import sys
from typing import Iterator, Tuple

try:
    # Package import (preferred)
    from . import sbgMessageParse
except Exception:  # pragma: no cover
    # Script-style import (backwards compatible)
    import sbgMessageParse  # type: ignore


SYNC1 = b'\xff'
SYNC2 = b'\x5a'


def iter_sbg_headers(
    connection: socket.socket,
    *,
    stop_event=None,
) -> Iterator[Tuple[bytes, bytes]]:
    """
    Yield (msg_id, msg_class) pairs from a raw SBG TCP byte stream.

    This matches the original byte-by-byte sync scan in the 2016 script.
    The caller is expected to pass the returned header bytes into
    `sbgMessageParse.parseSbgMessage(msg_class, msg_id, connection=connection, ...)`.
    """
    # Non-empty value to start the while loop
    byte = b'\x00'
    while byte:
        if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
            return

        try:
            # Receive one byte at a time
            byte = connection.recv(1)
        except socket.timeout:
            continue

        if not byte:
            return

        if byte != SYNC1:
            continue

        try:
            byte2 = connection.recv(1)
        except socket.timeout:
            continue

        if not byte2:
            return

        if byte2 != SYNC2:
            continue

        try:
            msg_id = connection.recv(1)
            msg_class = connection.recv(1)
        except socket.timeout:
            continue

        if not msg_id or not msg_class:
            return

        yield msg_id, msg_class


def main(bind: str = '0.0.0.0', port: int = 3001) -> None:  # pragma: no cover
    """Run the original standalone TCP decoder server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (bind, int(port))
    print('starting up on %s port %s' % server_address, file=sys.stderr)
    sock.bind(server_address)
    sock.listen(1)

    while True:
        print('waiting for a connection', file=sys.stderr)
        connection, client_address = sock.accept()
        try:
            print('connection from', client_address, file=sys.stderr)
            for msg_id, msg_class in iter_sbg_headers(connection):
                sbgMessageParse.parseSbgMessage(
                    msg_class,
                    msg_id,
                    connection=connection,
                    printFlag=True,
                    outputFile=sys.stdout,
                )
        finally:
            connection.close()


if __name__ == '__main__':  # pragma: no cover
    main()
