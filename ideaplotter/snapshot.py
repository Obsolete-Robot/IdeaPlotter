"""Prebuilt snapshots: build the game space once, let everyone else download it.

`export` (for whoever crawls Steam and runs `build`) packs data/space/ plus a slim games
table into out/snapshot/, with a manifest of sizes and checksums. Those files are attached
to a GitHub release. `download` fetches them into data/space/, after which `serve` works
without crawling Steam or computing a single embedding.

The slim table keeps what the map needs for every game at once (name, tags, review count)
and no store copy: short descriptions and About text are looked up per game when needed.
"""
import datetime as dt
import hashlib
import json
import shutil
import time
import urllib.request

import numpy as np

from . import config, steam
from .progress import Status, fmt_bytes, fmt_time

# Everything `Space` loads, except games.parquet, which `export` writes in its slim form.
FILES = ["emb.npy", "emb_meta.json", "nn_sim.npy", "nn_idx.npy", "probe.npz", "cluster.npy", "centroids.npy",
         "cluster_names.json", "xy.npy", "xyz.npy"]
OPTIONAL = ["calibration.json"]
GAME_COLUMNS = ["appid", "name", "tags", "review_count", "nsfw"]


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(out=None) -> None:
    from .space import load_aligned

    out = out or config.OUT / "snapshot"
    missing = [f for f in FILES if not (config.SPACE / f).exists()]
    if missing:
        raise SystemExit(f"cannot export: {', '.join(missing)} missing from {config.SPACE}; run `build` and `map3d` first")
    out.mkdir(parents=True, exist_ok=True)
    games, model = load_aligned()
    slim = games[GAME_COLUMNS].copy()
    # "furthest games" skips pages with almost no store text; keep that one fact about the dropped text
    slim["text_len"] = (games["short_description"].str.len() + games["about"].str.len()).astype(np.int32)
    slim.to_parquet(out / "games.parquet", index=False)
    names = ["games.parquet"]
    for f in FILES + [f for f in OPTIONAL if (config.SPACE / f).exists()]:
        shutil.copy2(config.SPACE / f, out / f)
        names.append(f)
    manifest = {
        "tag": config.SNAPSHOT_TAG,
        "created": dt.date.today().isoformat(),
        "model": model,
        "games": len(games),
        "files": {f: {"size": (out / f).stat().st_size, "sha256": _sha256(out / f)} for f in names},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = sum(v["size"] for v in manifest["files"].values())
    print(f"snapshot: {len(games):,} games, {len(names)} files, {total / 1e6:.0f} MB -> {out}")
    print(f"publish with:  gh release create {config.SNAPSHOT_TAG} {out.as_posix()}/* --title \"Game space snapshot\"")


def _fetch(url: str, dest, size: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": steam.USER_AGENT})
    part = dest.with_name(dest.name + ".part")
    done, t0, moved = 0, time.time(), time.time()

    def text() -> str:
        quiet = time.time() - moved
        speed = (f"no data for {fmt_time(quiet)}, still trying" if quiet >= 20
                 else f"{done / max(time.time() - t0, 0.1) / 1e6:.1f} MB/s")
        return f"{dest.name}: {fmt_bytes(done)} of {fmt_bytes(size)} ({done * 100 // max(size, 1)}%, {speed})"

    with Status(text), urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        while block := r.read(1 << 16):
            f.write(block)
            done += len(block)
            moved = time.time()
    print(f"  {dest.name}: {fmt_bytes(size)} in {fmt_time(time.time() - t0)}")
    part.replace(dest)


def download(url: str | None = None, force: bool = False) -> None:
    """Fetch the snapshot into data/space/, skipping files that are already there and intact."""
    base = (url or config.SNAPSHOT_URL).rstrip("/") + "/"
    req = urllib.request.Request(base + "manifest.json", headers={"User-Agent": steam.USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            manifest = json.loads(r.read().decode("utf-8"))
    except OSError as e:  # HTTP and connection errors
        raise SystemExit(f"Could not get the snapshot from {base} ({e}).\n"
                         "Check your internet connection. If this keeps happening, the release may have moved: "
                         "see the project's README.") from e
    files = manifest["files"]
    total = sum(v["size"] for v in files.values())
    print(f"snapshot {manifest.get('tag')}: {manifest['games']:,} games embedded with {manifest['model']}, "
          f"{total / 1e6:.0f} MB")
    config.SPACE.mkdir(parents=True, exist_ok=True)
    for name, info in files.items():
        dest = config.SPACE / name
        if dest.parent != config.SPACE:
            raise SystemExit(f"bad file name in manifest: {name!r}")
        if not force and dest.exists() and dest.stat().st_size == info["size"]:
            with Status(f"checking {name}, already on disk"):
                intact = _sha256(dest) == info["sha256"]
            if intact:
                print(f"  {name}: up to date")
                continue
        _fetch(base + name, dest, info["size"])
        with Status(f"checking {name}"):
            intact = _sha256(dest) == info["sha256"]
        if not intact:
            dest.unlink()
            raise SystemExit(f"{name} did not download correctly (checksum mismatch); run `download` again")
    (config.SPACE / "snapshot.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("snapshot ready.")
