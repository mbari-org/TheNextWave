#!/usr/bin/env python3

from __future__ import annotations

import csv
import gzip
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import shutil
import threading
from typing import Any


class RollingCsvLogger:
    def __init__(
        self,
        filename: str | os.PathLike[str],
        fieldnames: Sequence[str],
        *,
        extrasaction: str = 'raise',
        compresslevel: int = 6,
    ) -> None:
        self.path = Path(filename).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = tuple(fieldnames)
        self.extrasaction = extrasaction
        self.compresslevel = int(compresslevel)

        if not self.fieldnames:
            raise ValueError('fieldnames cannot be empty')

        if len(set(self.fieldnames)) != len(self.fieldnames):
            raise ValueError('fieldnames must be unique')

        if self.extrasaction not in ('raise', 'ignore'):
            raise ValueError("extrasaction must be 'raise' or 'ignore'")

        if not 0 <= self.compresslevel <= 9:
            raise ValueError('compresslevel must be between 0 and 9')

        self._lock = threading.RLock()
        self._closed = False
        self._file = None
        self._writer = None

        self._validate_existing_header()
        self._open_file()

    def _validate_existing_header(self) -> None:
        try:
            if self.path.stat().st_size == 0:
                return
        except FileNotFoundError:
            return

        with self.path.open(
            'r',
            encoding='utf-8',
            newline='',
        ) as file:
            try:
                existing_header = tuple(next(csv.reader(file)))
            except StopIteration:
                return

        if existing_header != self.fieldnames:
            raise ValueError(
                'Existing CSV header does not match configured fieldnames:\n'
                f'existing:   {existing_header}\n'
                f'configured: {self.fieldnames}'
            )

    def _open_file(self) -> None:
        has_content = self.path.exists() and self.path.stat().st_size > 0

        self._file = self.path.open(
            'a',
            encoding='utf-8',
            newline='',
        )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.fieldnames,
            extrasaction=self.extrasaction,
            restval='',
            lineterminator='\n',
        )

        if not has_content:
            self._writer.writeheader()
            self._file.flush()

    def _close_file(self) -> None:
        if self._file is None:
            return

        self._file.flush()
        self._file.close()

        self._file = None
        self._writer = None

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return ''.join(f'\\x{byte:02x}' for byte in bytes(value))

        return value

    def _normalize_row(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            key: self._normalize_value(value)
            for key, value in row.items()
        }

    def _archive_path(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime(
            '%Y%m%dT%H%M%S.%fZ'
        )

        sequence = 0

        while True:
            suffix = '' if sequence == 0 else f'.{sequence}'

            archive = self.path.with_name(
                f'{self.path.stem}.{timestamp}'
                f'{suffix}{self.path.suffix}'
            )

            compressed = Path(f'{archive}.gz')

            if not archive.exists() and not compressed.exists():
                return archive

            sequence += 1

    def write(self, row: Mapping[str, Any]) -> None:
        if not isinstance(row, Mapping):
            raise TypeError('row must be a mapping')

        with self._lock:
            if self._closed:
                raise RuntimeError('logger is closed')

            assert self._writer is not None
            assert self._file is not None

            self._writer.writerow(self._normalize_row(row))
            self._file.flush()

    def roll(self) -> Path | None:
        with self._lock:
            if self._closed:
                raise RuntimeError('logger is closed')

            self._close_file()

            archive = None
            compressed = None
            temporary = None

            try:
                if self.path.exists() and self.path.stat().st_size > 0:
                    archive = self._archive_path()
                    compressed = Path(f'{archive}.gz')
                    temporary = Path(f'{compressed}.tmp')

                    os.replace(self.path, archive)

                    try:
                        with archive.open('rb') as source:
                            with gzip.open(
                                temporary,
                                'wb',
                                compresslevel=self.compresslevel,
                            ) as destination:
                                shutil.copyfileobj(source, destination)

                        os.replace(temporary, compressed)
                        archive.unlink()

                    except Exception:
                        try:
                            temporary.unlink()
                        except FileNotFoundError:
                            pass

                        raise

            finally:
                self._open_file()

            return compressed

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                return

            assert self._file is not None
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            self._closed = True
            self._close_file()

    def __enter__(self) -> RollingCsvLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
