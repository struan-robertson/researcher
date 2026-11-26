"""Content-addressed managed artefact files."""

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_ARTEFACT_ID = re.compile(r"[0-9a-f]{64}")
_TEMPORARY_PREFIX = ".partial-"


@dataclass(frozen=True)
class Blob:
    """A completed file in the artefact store."""

    artefact_id: str
    byte_size: int


class ArtefactStore:
    """SHA-256 addressed files below one directory.

    The digest names each file, so caller-supplied filenames and paths never
    become storage names, identical bytes share one file and completed files
    are never rewritten.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _stage(self) -> Iterator[Path]:
        """Yield a unique empty file path inside the store for content being written.

        Leaving the context removes any file still at the path, including after
        a failed write. Staging files left by a killed process are not
        collected.
        """
        descriptor, name = tempfile.mkstemp(
            dir=self.root.resolve(), prefix=_TEMPORARY_PREFIX
        )
        os.close(descriptor)
        staged = Path(name)
        try:
            yield staged
        finally:
            staged.unlink(missing_ok=True)

    def _publish(self, staged: Path) -> Blob:
        """Move a completed staging file to its digest path, then sync the store.

        The file is fsynced before the move, so the digest name never refers to
        an incomplete blob.
        """
        with staged.open("rb+") as handle:
            artefact_id = hashlib.file_digest(handle, "sha256").hexdigest()
            byte_size = os.fstat(handle.fileno()).st_size
            os.fsync(handle.fileno())
        completed = self.root / artefact_id
        if completed.exists():
            staged.unlink()
        else:
            os.replace(staged, completed)
        # Existing blobs may come from a previous directory-sync failure.
        _fsync_directory(self.root)
        return Blob(artefact_id, byte_size)

    def put_bytes(self, data: bytes) -> Blob:
        with self._stage() as staged:
            staged.write_bytes(data)
            return self._publish(staged)

    def put_file(self, source: Path) -> Blob:
        with self._stage() as staged:
            shutil.copyfile(source, staged)
            return self._publish(staged)

    def path(self, artefact_id: str) -> Path:
        if _ARTEFACT_ID.fullmatch(artefact_id) is None:
            raise ValueError(f"not a SHA-256 artefact ID: {artefact_id!r}")
        return self.root / artefact_id


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
