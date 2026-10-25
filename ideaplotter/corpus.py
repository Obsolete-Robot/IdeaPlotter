"""Turn raw Steam store items into a clean table and one embedding document per game."""
import html
import json
import re

import pandas as pd

from . import config, steam

ABOUT_CHARS = 800  # store copy beyond this is mostly feature lists, system notes and calls to wishlist

_DROP_BLOCKS = re.compile(
    r"\[(img|video|previewyoutube|carousel|dynamiclink)[^\]]*\].*?\[/\1\]", re.I | re.S
)
_BULLET = re.compile(r"\[\*\]")
_BREAK = re.compile(r"\[/?(p|br|h[1-6]|list|olist|hr)[^\]]*\]", re.I)
_TAG = re.compile(r"\[/?[a-z0-9]+(=[^\]]*)?\]", re.I)
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|\{STEAM_[A-Z_]+\}\S*)")
_SPACES = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\s*\n\s*")


def clean_bbcode(text: str) -> str:
    if not text:
        return ""
    text = _DROP_BLOCKS.sub(" ", text)
    text = _BULLET.sub("\n- ", text)
    text = _BREAK.sub("\n", text)
    text = _TAG.sub("", text)
    text = _HTML.sub(" ", text)
    text = _URL.sub("", text)
    text = html.unescape(text).replace("\r", "")
    text = _SPACES.sub(" ", text)
    text = _NEWLINES.sub("\n", text)
    return text.strip()


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c < "ɐ" for c in letters) / len(letters)


def doc_text(name: str, short: str, about: str, tags: list[str]) -> str:
    """The exact text that gets embedded. Generated ideas are embedded in this same shape.

    Tags go before the long description so that truncation at the model's token limit
    only ever cuts store copy, never the metadata."""
    about = about[:ABOUT_CHARS].rsplit(" ", 1)[0] if len(about) > ABOUT_CHARS else about
    parts = [name.strip(), short.strip(), "Tags: " + ", ".join(tags), about.strip()]
    return "\n".join(p for p in parts if p)


def _names(objs):
    return [o.get("name", "") for o in objs or [] if o.get("name")]


def parse_item(it: dict, tagnames: dict) -> dict:
    bi = it.get("basic_info") or {}
    tags = [tagnames.get(str(t["tagid"])) for t in sorted(it.get("tags") or [], key=lambda t: -t.get("weight", 0))]
    tags = [t for t in tags if t]
    rev = (it.get("reviews") or {}).get("summary_filtered") or {}
    rel = it.get("release") or {}
    price = (it.get("best_purchase_option") or {}).get("final_price_in_cents")
    cats = it.get("categories") or {}
    return {
        "appid": int(it["appid"]),
        "name": (it.get("name") or "").strip(),
        "short_description": clean_bbcode(bi.get("short_description", "")),
        "about": clean_bbcode(it.get("full_description_bbcode", "")),
        "tags": tags,
        "category_ids": sorted(set(cats.get("supported_player_categoryids", []) + cats.get("feature_categoryids", []))),
        "developers": _names(bi.get("developers")),
        "publishers": _names(bi.get("publishers")),
        "franchises": _names(bi.get("franchises")),
        "release_date": rel.get("steam_release_date") or rel.get("original_release_date"),
        "coming_soon": bool(rel.get("is_coming_soon")),
        "early_access": bool(it.get("is_early_access")),
        "is_free": bool(it.get("is_free")),
        "price_cents": int(price) if price else None,
        "review_count": rev.get("review_count", 0),
        "percent_positive": rev.get("percent_positive"),
        "store_slug": it.get("store_url_path", ""),
    }


def build(min_text: int = 120, min_tags: int = 3, min_latin: float = 0.7) -> pd.DataFrame:
    """Parse cached pages, filter out stubs and non-English pages, and write data/games.parquet."""
    tagnames = json.loads(config.TAGS.read_text(encoding="utf-8"))
    rows = [parse_item(it, tagnames) for it in steam.iter_items() if it.get("success") == 1 and it.get("name")]
    df = pd.DataFrame(rows)
    n_all = len(df)

    text = df["short_description"] + " " + df["about"]
    df["latin"] = text.map(latin_ratio)
    keep = (text.str.len() >= min_text) & (df["tags"].map(len) >= min_tags) & (df["latin"] >= min_latin)
    df = df[keep].reset_index(drop=True)
    df["nsfw"] = df["tags"].map(lambda ts: bool(config.NSFW_TAGS.intersection(ts[:10])))
    df["doc"] = [doc_text(*r) for r in zip(df["name"], df["short_description"], df["about"], df["tags"])]

    df.to_parquet(config.GAMES, index=False)
    print(f"corpus: kept {len(df):,} of {n_all:,} games "
          f"(dropped stubs, <{min_tags} tags, or <{min_latin:.0%} Latin-script text)")
    return df


def load() -> pd.DataFrame:
    return pd.read_parquet(config.GAMES)
