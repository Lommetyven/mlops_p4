import shutil
import tarfile
from argparse import ArgumentParser
from pathlib import Path


def ensure_inside_workspace(path):
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    if workspace != resolved and workspace not in resolved.parents:
        raise ValueError(f"Path must stay inside workspace: {resolved}")
    return resolved


def pack(source, output, allow_empty=False):
    source = ensure_inside_workspace(Path(source))
    output = ensure_inside_workspace(Path(output))

    if not source.exists():
        if not allow_empty:
            raise FileNotFoundError(source)
        source.mkdir(parents=True, exist_ok=True)

    if source.is_dir() and not allow_empty and not any(source.iterdir()):
        raise ValueError(f"Source directory is empty: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        archive.add(source, arcname=source.name)

    print(f"Packed {source} -> {output}")


def unpack(archive, output, clean=False):
    archive = ensure_inside_workspace(Path(archive))
    output = ensure_inside_workspace(Path(output))

    if not archive.exists():
        raise FileNotFoundError(archive)

    if clean and output.exists():
        if output == Path.cwd().resolve():
            raise ValueError("Refusing to clean workspace root.")
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        safe_extract(tar, output)

    print(f"Unpacked {archive} -> {output}")


def safe_extract(archive, output):
    output = output.resolve()
    for member in archive.getmembers():
        target = (output / member.name).resolve()
        if output != target and output not in target.parents:
            raise ValueError(f"Archive member escapes output directory: {member.name}")
    archive.extractall(output, filter="data")


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack")
    pack_parser.add_argument("--source", required=True)
    pack_parser.add_argument("--output", required=True)
    pack_parser.add_argument("--allow-empty", action="store_true")

    unpack_parser = subparsers.add_parser("unpack")
    unpack_parser.add_argument("--archive", required=True)
    unpack_parser.add_argument("--output", required=True)
    unpack_parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()
    if args.command == "pack":
        pack(args.source, args.output, allow_empty=args.allow_empty)
    else:
        unpack(args.archive, args.output, clean=args.clean)


if __name__ == "__main__":
    main()
