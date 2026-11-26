import hashlib
import os
import stat
from pathlib import Path

import pytest

from researcher import artefacts
from researcher.artefacts import ArtefactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtefactStore:
    return ArtefactStore(tmp_path / "artefacts")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_identical_bytes_share_one_file(store: ArtefactStore) -> None:
    first = store.put_bytes(b"same content")
    second = store.put_bytes(b"same content")
    assert first == second
    assert first.artefact_id == digest(b"same content")
    assert first.byte_size == len(b"same content")
    assert [path.name for path in store.root.iterdir()] == [first.artefact_id]
    assert store.path(first.artefact_id).read_bytes() == b"same content"


def test_different_bytes_are_stored_separately(store: ArtefactStore) -> None:
    first = store.put_bytes(b"first")
    second = store.put_bytes(b"second")
    assert first.artefact_id != second.artefact_id
    assert {path.name for path in store.root.iterdir()} == {
        first.artefact_id,
        second.artefact_id,
    }


def test_empty_content_is_a_valid_artefact(store: ArtefactStore) -> None:
    blob = store.put_bytes(b"")
    assert blob.artefact_id == digest(b"")
    assert blob.byte_size == 0
    assert store.path(blob.artefact_id).read_bytes() == b""


def test_filenames_are_not_identity(store: ArtefactStore, tmp_path: Path) -> None:
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "paper.pdf"
    first.write_bytes(b"%PDF first")
    second = second_dir / "paper.pdf"
    second.write_bytes(b"%PDF second")
    copy = second_dir / "renamed.pdf"
    copy.write_bytes(b"%PDF first")
    assert store.put_file(first).artefact_id != store.put_file(second).artefact_id
    assert store.put_file(first).artefact_id == store.put_file(copy).artefact_id
    stored = {path.name for path in store.root.iterdir()}
    assert stored == {digest(b"%PDF first"), digest(b"%PDF second")}
    assert "paper.pdf" not in stored


def test_import_does_not_modify_the_source(
    store: ArtefactStore, tmp_path: Path
) -> None:
    source = tmp_path / "downloaded.pdf"
    source.write_bytes(b"%PDF unchanged")
    before = source.stat()
    store.put_file(source)
    after = source.stat()
    assert source.read_bytes() == b"%PDF unchanged"
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_size == before.st_size


def test_missing_source_leaves_no_temporary_file(
    store: ArtefactStore, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        store.put_file(tmp_path / "absent.pdf")
    assert list(store.root.iterdir()) == []


def test_replace_failure_removes_the_temporary_file(
    store: ArtefactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(artefacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.put_bytes(b"%PDF interrupted")
    assert list(store.root.iterdir()) == []


@pytest.mark.parametrize(
    "artefact_id",
    [
        "",
        "../../etc/passwd",
        "0" * 63,
        "0" * 65,
        "A" * 64,
        "../" + "0" * 61,
        "0" * 63 + "/",
    ],
)
def test_paths_must_be_sha256_digests(store: ArtefactStore, artefact_id: str) -> None:
    with pytest.raises(ValueError, match="not a SHA-256 artefact ID"):
        store.path(artefact_id)


def test_paths_stay_inside_the_store(store: ArtefactStore) -> None:
    path = store.path(digest(b"content"))
    assert path.parent == store.root
    assert path.name == digest(b"content")


def test_directory_is_synced_after_install(
    store: ArtefactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[Path] = []
    real_sync = artefacts._fsync_directory

    def record_sync(path: Path) -> None:
        synced.append(path)
        real_sync(path)

    monkeypatch.setattr(artefacts, "_fsync_directory", record_sync)
    store.put_bytes(b"synced content")
    assert synced == [store.root]


def test_directory_sync_failure_keeps_completed_blob(
    store: ArtefactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"%PDF-1.7 unsynced"

    def fail_sync(path: Path) -> None:
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(artefacts, "_fsync_directory", fail_sync)
    with pytest.raises(OSError, match="simulated"):
        store.put_bytes(data)
    completed = store.path(digest(data))
    assert completed.read_bytes() == data
    assert [path.name for path in store.root.iterdir()] == [digest(data)]


def test_existing_blob_is_resynced(
    store: ArtefactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"%PDF-1.7 retried"
    real_sync = artefacts._fsync_directory

    def fail_sync(path: Path) -> None:
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(artefacts, "_fsync_directory", fail_sync)
    with pytest.raises(OSError, match="simulated"):
        store.put_bytes(data)

    synced: list[Path] = []

    def record_sync(path: Path) -> None:
        synced.append(path)
        real_sync(path)

    monkeypatch.setattr(artefacts, "_fsync_directory", record_sync)
    blob = store.put_bytes(data)
    assert synced == [store.root]
    assert store.path(blob.artefact_id).read_bytes() == data
    assert [path.name for path in store.root.iterdir()] == [digest(data)]


def test_existing_blob_is_not_rewritten(store: ArtefactStore) -> None:
    blob = store.put_bytes(b"%PDF-1.7 intact")
    path = store.path(blob.artefact_id)
    before = path.stat()
    store.put_bytes(b"%PDF-1.7 intact")
    after = path.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_import_sync_failure_keeps_source_and_completed_blob(
    store: ArtefactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"%PDF source intact"
    source = tmp_path / "downloaded.pdf"
    source.write_bytes(data)

    def fail_sync(path: Path) -> None:
        raise OSError("simulated directory sync failure")

    monkeypatch.setattr(artefacts, "_fsync_directory", fail_sync)
    with pytest.raises(OSError, match="simulated"):
        store.put_file(source)
    assert source.read_bytes() == data
    assert store.path(digest(data)).read_bytes() == data
    assert [path.name for path in store.root.iterdir()] == [digest(data)]


def record_closed_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, bool]]:
    """Record each closed descriptor with whether it named a directory."""
    closed: list[tuple[int, bool]] = []
    real_close = artefacts.os.close

    def record_close(descriptor: int) -> None:
        closed.append((descriptor, stat.S_ISDIR(os.fstat(descriptor).st_mode)))
        real_close(descriptor)

    monkeypatch.setattr(artefacts.os, "close", record_close)
    return closed


def test_directory_descriptor_is_closed(
    store: ArtefactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = record_closed_descriptors(monkeypatch)
    store.put_bytes(b"descriptor")
    [descriptor] = [fd for fd, is_directory in closed if is_directory]
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_directory_descriptor_is_closed_on_sync_failure(
    store: ArtefactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = record_closed_descriptors(monkeypatch)
    real_fsync = artefacts.os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated directory sync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artefacts.os, "fsync", fail_directory_sync)
    with pytest.raises(OSError, match="simulated"):
        store.put_bytes(b"descriptor")
    [descriptor] = [fd for fd, is_directory in closed if is_directory]
    with pytest.raises(OSError):
        os.fstat(descriptor)
