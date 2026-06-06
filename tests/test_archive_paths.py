from pathlib import Path

from scripts.archive_paths import pack, unpack


def test_pack_and_unpack_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("data/raw")
    source.mkdir(parents=True)
    (source / "sample.txt").write_text("hello", encoding="utf-8")

    archive = Path("data/dvc_archives/raw.tar.gz")
    restored = Path("restored")

    pack(source, archive)
    unpack(archive, restored)

    assert archive.exists()
    assert (restored / "raw" / "sample.txt").read_text(encoding="utf-8") == "hello"


def test_pack_allows_empty_directory_when_requested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path("models")
    archive = Path("data/dvc_archives/models.tar.gz")

    pack(source, archive, allow_empty=True)

    assert archive.exists()
