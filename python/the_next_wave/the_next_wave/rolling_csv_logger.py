from __future__ import annotations

import csv
import io
import logging
import os
import threading
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from concurrent_log_handler import ConcurrentTimedRotatingFileHandler


def _encode_csv_row(values: Sequence[Any]) -> str:
    """
    Encode one CSV row without the final newline.
    """
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerow(values)
    return output.getvalue().removesuffix("\n")


class DictCsvFormatter(logging.Formatter):
    def __init__(
        self,
        fieldnames: Sequence[str],
        *,
        extrasaction: str = "raise",
    ) -> None:
        super().__init__()
        self.fieldnames = tuple(fieldnames)
        self.extrasaction = extrasaction

    def format(self, record: logging.LogRecord) -> str:
        row = getattr(record, "csv_row", None)

        if not isinstance(row, Mapping):
            raise TypeError("csv_row must be a mapping")

        output = io.StringIO(newline="")

        writer = csv.DictWriter(
            output,
            fieldnames=self.fieldnames,
            extrasaction=self.extrasaction,
            restval="",
            lineterminator="\n",
        )
        writer.writerow(row)

        return output.getvalue().removesuffix("\n")


class HeaderTimedRotatingFileHandler(
    ConcurrentTimedRotatingFileHandler
):
    """
    Concurrent timed rotation with:

    - gzip compression;
    - unlimited archive retention;
    - a CSV header in every file;
    - timestamped archive names.
    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        *,
        csv_header: str,
        when: str = "H",
        interval: int = 1,
        utc: bool = True,
        max_bytes: int = 0,
    ) -> None:
        self.csv_header = csv_header

        super().__init__(
            filename=filename,
            when=when,
            interval=interval,

            # For the timed handler, zero means do not delete archives.
            backupCount=0,

            encoding="utf-8",
            delay=True,
            utc=utc,

            # Zero means time-only rotation. A positive value adds
            # size-based rotation in addition to timed rotation.
            maxBytes=max_bytes,

            use_gzip=True,
            newline="",
            terminator="\n",
        )

    def finalize_handler_configuration(self) -> None:
        """
        Configure archive names before rollover initialization.

        Example UTC filename:
            vehicle.20260730T050000Z.csv.gz
        """
        super().finalize_handler_configuration()

        if self.utc:
            self.suffix = "%Y%m%dT%H%M%SZ"
        else:
            self.suffix = "%Y%m%dT%H%M%S"

        self.namer = self._archive_namer

    def _archive_namer(self, default_name: str) -> str:
        """
        Convert:

            vehicle.csv.20260730T050000Z

        into:

            vehicle.20260730T050000Z.csv

        Gzip subsequently adds .gz.
        """
        prefix = self.baseFilename + "."

        if not default_name.startswith(prefix):
            return default_name

        timestamp = default_name[len(prefix):]
        base = Path(self.baseFilename)

        return str(
            base.with_name(
                f"{base.stem}.{timestamp}{base.suffix}"
            )
        )

    def _write_header_if_empty_locked(self) -> None:
        """
        Write the header if the active file is absent or empty.

        The concurrent handler lock must already be held.
        """
        try:
            empty = os.path.getsize(self.baseFilename) == 0
        except FileNotFoundError:
            empty = True

        if empty:
            self.clh.do_write(self.csv_header)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Rotate if necessary, then atomically write the header and row.
        """
        try:
            msg = self.format(record)

            try:
                self.clh._do_lock()
                self.clh._check_stream()

                try:
                    if self.shouldRollover(record):
                        self.doRollover()
                except Exception as exc:
                    self._console_log(
                        f"Unable to do rollover: {exc}\n"
                        f"{traceback.format_exc()}"
                    )

                # This happens after rollover but before the first data row.
                self._write_header_if_empty_locked()
                self.clh.do_write(msg)

            finally:
                self.clh._do_unlock()

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class RollingCsvLogger:
    def __init__(
        self,
        filename: str | os.PathLike[str],
        fieldnames: Sequence[str],
        *,
        when: str = "H",
        interval: int = 1,
        utc: bool = True,
        max_bytes: int = 0,
        extrasaction: str = "raise",
    ) -> None:
        fields = tuple(fieldnames)

        if not fields:
            raise ValueError("fieldnames cannot be empty")

        if any(
            not isinstance(field, str) or not field
            for field in fields
        ):
            raise ValueError(
                "every field name must be a non-empty string"
            )

        if len(set(fields)) != len(fields):
            raise ValueError("fieldnames must be unique")

        if extrasaction not in {"raise", "ignore"}:
            raise ValueError(
                "extrasaction must be 'raise' or 'ignore'"
            )

        if interval < 1:
            raise ValueError("interval must be at least 1")

        if max_bytes < 0:
            raise ValueError("max_bytes cannot be negative")

        self.path = Path(filename).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = fields
        self._closed = False
        self._lock = threading.RLock()

        self._validate_existing_header()

        self._handler = HeaderTimedRotatingFileHandler(
            self.path,
            csv_header=_encode_csv_row(self.fieldnames),
            when=when,
            interval=interval,
            utc=utc,
            max_bytes=max_bytes,
        )

        self._handler.setFormatter(
            DictCsvFormatter(
                self.fieldnames,
                extrasaction=extrasaction,
            )
        )

        # Standalone logger avoids conflicts with globally registered
        # logging.Logger instances.
        self._logger = logging.Logger(
            name=f"rolling-csv:{self.path}",
            level=logging.INFO,
        )
        self._logger.propagate = False
        self._logger.addHandler(self._handler)

    def _validate_existing_header(self) -> None:
        """
        Refuse to append if the existing active CSV has another schema.
        """
        try:
            if self.path.stat().st_size == 0:
                return
        except FileNotFoundError:
            return

        with self.path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            existing_header = tuple(next(csv.reader(file)))

        if existing_header != self.fieldnames:
            raise ValueError(
                "Existing CSV header does not match fieldnames:\n"
                f"existing:   {existing_header}\n"
                f"configured: {self.fieldnames}"
            )

    def write(self, row: Mapping[str, Any]) -> None:
        if not isinstance(row, Mapping):
            raise TypeError("row must be a mapping")

        with self._lock:
            if self._closed:
                raise RuntimeError("logger is closed")

            self._logger.info(
                "",
                extra={"csv_row": dict(row)},
            )

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._handler.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            self._closed = True
            self._logger.removeHandler(self._handler)
            self._handler.close()

    def __enter__(self) -> RollingCsvLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
