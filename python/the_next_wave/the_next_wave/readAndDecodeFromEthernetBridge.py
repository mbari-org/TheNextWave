"""
Ethernet bridge reader / decoder.

Provides a small iterator that scans for SBG sync bytes and yields message
headers so callers can delegate payload parsing to `sbgMessageParse`.
"""

from __future__ import annotations

import socket
import sys
import time
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
    reader,
    *,
    stop_event=None,
) -> Iterator[Tuple[bytes, bytes]]:
    """
    Yield (msg_id, msg_class) pairs from a buffered SBG byte stream.

    `reader` is a blocking binary file object -- `sock.makefile('rb')` for a
    socket -- not a raw socket. Reading through a BufferedReader costs one
    syscall per buffer fill rather than one per byte, which matters most during
    a backfill burst and during sync-byte scans after a framing error.

    Blocking is deliberate. A peer that vanishes without a FIN is detected by
    TCP keepalive, which tears the socket down and makes `read` return b''
    here; a read timeout cannot tell "silent between bursts" from "gone", and
    firing mid-message can leave the BufferedReader's internal buffer in an
    inconsistent state. Shutting the socket down from another thread also
    unblocks these reads, which is how stop() interrupts us.

    The caller passes the returned header bytes into
    `sbgMessageParse.parseSbgMessage(msg_class, msg_id, connection=reader, ...)`.
    """
    # Non-empty value to start the while loop
    byte = b'\x00'
    while byte:
        if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
            return

        byte = reader.read(1)
        if not byte:
            return

        if byte != SYNC1:
            continue

        byte2 = reader.read(1)
        if not byte2:
            return

        if byte2 != SYNC2:
            continue

        msg_id = reader.read(1)
        msg_class = reader.read(1)
        if not msg_id or not msg_class:
            return

        yield msg_id, msg_class


def main(bind: str = '0.0.0.0', port: int = 3002) -> None:  # pragma: no cover
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
    import sys
    main(bind=sys.argv[1], port=int(sys.argv[2]))
