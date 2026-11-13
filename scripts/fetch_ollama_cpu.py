"""Fetch only CPU runtime members from the official release ZIP using HTTP ranges.

Uses Windows curl for the platform TLS trust store. zipfile validates member CRCs;
this is not a claim that a partial download validates the whole-archive SHA256.
"""
from __future__ import annotations

import io
import json
import subprocess
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str, length: int):
        self.url, self.length, self.position = url, length, 0
        self.cached_start, self.cached_data = 0, b""

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        return self.position

    def cache(self, start, count):
        end = min(start + count, self.length) - 1
        if end < start:
            return
        args = ["curl.exe", "-fsSL", "--retry", "2", "--retry-all-errors", "--connect-timeout", "15", "--max-time", "300",
                "--max-filesize", str(end - start + 1), "-r", f"{start}-{end}", self.url]
        data = subprocess.run(args, check=True, capture_output=True).stdout
        if len(data) != end - start + 1:
            raise RuntimeError("Server did not honor the requested ZIP byte range")
        self.cached_start, self.cached_data = start, data

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b""
        offset = self.position - self.cached_start
        if offset < 0 or offset + size > len(self.cached_data):
            self.cache(self.position, max(size, 65536))
            offset = 0
        data = self.cached_data[offset:offset + size]
        self.position += len(data)
        return data


def main():
    release = json.loads((ROOT / ".runtime/ollama-release.json").read_text())
    asset = next(a for a in release["assets"] if a["name"] == "ollama-windows-amd64.zip")
    remote = RemoteZip(asset["browser_download_url"], asset["size"])
    destination = (ROOT / ".runtime/ollama").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(remote) as archive:
        members = [i for i in archive.infolist() if not i.is_dir()]
        (ROOT / ".runtime/ollama-members.json").write_text(json.dumps(
            [{"name":i.filename,"bytes":i.file_size,"compressed":i.compress_size} for i in members], indent=2))
        chosen = [i for i in members if i.filename.replace("\\", "/") == "ollama.exe"
                  or (i.filename.replace("\\", "/").startswith("lib/ollama/")
                      and not any(s in i.filename.lower() for s in ("cuda", "rocm", "vulkan", "mlx")))]
        for info in chosen:
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe ZIP member")
            if target.exists() and target.stat().st_size == info.file_size:
                if zlib.crc32(target.read_bytes()) == info.CRC:
                    continue
            print(f"Fetching {info.filename}: {info.compress_size / 2**20:.1f} MiB", flush=True)
            remote.cache(info.header_offset, info.compress_size + len(info.filename.encode()) + 4096)
            contents = archive.read(info)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
    if not (destination / "ollama.exe").exists():
        raise RuntimeError("CPU selection did not contain ollama.exe")
    (destination / "release.json").write_text(json.dumps({
        "tag": release["tag_name"], "source":asset["browser_download_url"],
        "verification":"HTTPS official release; zipfile member CRC; not whole-archive SHA256",
        "members":[i.filename for i in chosen]}, indent=2))


if __name__ == "__main__":
    main()
