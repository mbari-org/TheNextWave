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
    connection: socket.socket,
    *,
    stop_event=None,
    idle_timeout_sec: float = 0.0,
) -> Iterator[Tuple[bytes, bytes]]:
    """
    Yield (msg_id, msg_class) pairs from a raw SBG TCP byte stream.

    This matches the original byte-by-byte sync scan in the 2016 script.
    The caller is expected to pass the returned header bytes into
    `sbgMessageParse.parseSbgMessage(msg_class, msg_id, connection=connection, ...)`.

    `idle_timeout_sec` > 0 makes the iterator give up after that long without
    receiving a single byte. A peer that disappears without a FIN/RST (power
    cut, pulled cable, bridge reboot, NAT expiry) leaves recv timing out
    forever, so without this the caller never regains control and the listening
    socket never calls accept() again -- the SWIFT can then never reconnect.
    """
    # Non-empty value to start the while loop
    byte = b'\x00'
    last_rx = time.monotonic()

    def idle_expired() -> bool:
        return (
            idle_timeout_sec > 0.0
            and (time.monotonic() - last_rx) > idle_timeout_sec
        )

    while byte:
        if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
            return

        try:
            # Receive one byte at a time
            byte = connection.recv(1)
        except socket.timeout:
            if idle_expired():
                return
            continue

        if byte:
            last_rx = time.monotonic()

        if not byte:
            return

        if byte != SYNC1:
            continue

        try:
            byte2 = connection.recv(1)
        except socket.timeout:
            if idle_expired():
                return
            continue

        if not byte2:
            return
        last_rx = time.monotonic()

        if byte2 != SYNC2:
            continue

        try:
            msg_id = connection.recv(1)
            msg_class = connection.recv(1)
        except socket.timeout:
            if idle_expired():
                return
            continue

        if not msg_id or not msg_class:
            return
        last_rx = time.monotonic()

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
