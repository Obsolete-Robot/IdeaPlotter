"""Paths and defaults shared by every stage of the pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """KEY=value lines from .env (written by the installer); variables already set win."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
RAW = DATA / "raw"          # gzipped Steam API pages, one per 1000 games (only for building a snapshot)
STEAM_CACHE = DATA / "steam_cache"  # store items looked up one game at a time, one JSON file per appid
SPACE = Path(os.environ.get("IDEAPLOTTER_SPACE") or DATA / "space")  # embeddings + derived geometry
ADDED = SPACE / "added"     # games looked up on Steam and added to the map after the snapshot was built
OUT = ROOT / "out"          # human-facing results

GAMES = DATA / "games.parquet"  # full table with store text, written by `build`
SNAPSHOT_GAMES = SPACE / "games.parquet"  # slim table shipped with a snapshot (no store copy)
TAGS = DATA / "tags.json"
CATEGORIES = DATA / "categories.json"

# Multilingual, strong on clustering benchmarks, small enough for an 11 GB GPU.
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_MAX_TOKENS = 512

LLM_MODEL = "claude-opus-5-5"
# default writer on OpenRouter: in a side-by-side test it gave the most original, specific ideas
# (about $0.03 per idea, ~20 s). For cheap bulk runs use --model anthropic/claude-haiku-5.5 (~$0.002).
OPENROUTER_MODEL = "anthropic/claude-sonnet-5.5"

# The prebuilt snapshot `download` fetches: manifest.json and the files it lists, attached to
# this GitHub release. Bump the tag when a new snapshot is published.
SNAPSHOT_TAG = "snapshot-2026-10-21"
SNAPSHOT_URL = (os.environ.get("IDEAPLOTTER_SNAPSHOT_URL")
                or f"https://github.com/Obsolete-Robot/IdeaPlotter/releases/download/{SNAPSHOT_TAG}/")

# Tags that mark adult-only titles. These games stay in the corpus (so the space they
# occupy counts as "taken"), but they are never used as blend anchors.
NSFW_TAGS = {"Sexual Content", "Nudity", "NSFW", "Hentai", "Mature"}
