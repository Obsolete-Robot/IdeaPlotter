"""Steam's public store APIs: the full catalog crawl, and polite one-game lookups.

The crawl (`fetch`) is for whoever builds a snapshot. IStoreQueryService/Query needs no
API key and returns 1000 store items per call, including the full store description and
weighted user tags. Sorting by appid (sort=2) keeps pagination stable while new games are
being added, and pages are cached on disk so an interrupted crawl resumes where it stopped.

Everyone else runs on a downloaded snapshot and only asks Steam about the games they open
(`get_items`, `search`): one request at a time, at most one per second, and every game is
cached on disk so it is fetched at most once a month.
"""
import gzip
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

API = "https://api.steampowered.com"
PAGE_SIZE = 1000
SORT_BY_APPID = 2
USER_AGENT = "IdeaPlotter/0.1 (semantic game-space research)"
CONTEXT = {"language": "english", "country_code": "US", "steam_realm": 1}

DATA_REQUEST = {
    "include_basic_info": True,
    "include_tag_count": 20,
    "include_release": True,
    "include_full_description": True,
    "include_reviews": True,
    "include_platforms": True,
}


def _get(path: str, params: dict, retries: int = 8) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        wait = min(120, 5 * 2 ** attempt)
        print(f"  request failed ({err}); retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"giving up on {path} after {retries} attempts")


def _query_page(start: int) -> dict:
    body = {
        "query": {
            "start": start,
            "count": PAGE_SIZE,
            "sort": SORT_BY_APPID,
            "filters": {"type_filters": {"include_games": True}},
        },
        "context": CONTEXT,
        "data_request": DATA_REQUEST,
    }
    return _get("IStoreQueryService/Query/v1/", {"input_json": json.dumps(body, separators=(",", ":"))})["response"]


def fetch_vocab() -> None:
    tags = _get("IStoreService/GetTagList/v1/", {"language": "english"})["response"]["tags"]
    config.TAGS.write_text(json.dumps({str(t["tagid"]): t["name"] for t in tags}, indent=0), encoding="utf-8")
    cats = _get("IStoreBrowseService/GetStoreCategories/v1/", {"language": "english"})["response"]["categories"]
    config.CATEGORIES.write_text(
        json.dumps({str(c["categoryid"]): c["display_name"] for c in cats}, indent=0), encoding="utf-8"
    )
    print(f"vocab: {len(tags)} tags, {len(cats)} categories")


def fetch(delay: float = 1.0, refresh: bool = False) -> None:
    """Download every page of the catalog into data/raw/. Existing pages are skipped unless refresh."""
    config.RAW.mkdir(parents=True, exist_ok=True)
    fetch_vocab()

    total = _query_page(0)["metadata"]["total_matching_records"]
    starts = range(0, total, PAGE_SIZE)
    print(f"catalog: {total:,} game entries -> {len(starts)} pages")
    t0 = time.time()
    for n, start in enumerate(starts, 1):
        path = config.RAW / f"page_{start:07d}.json.gz"
        if path.exists() and not refresh:
            continue
        page = _query_page(start)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(page.get("store_items", []), f, separators=(",", ":"))
        rate = n / (time.time() - t0)
        print(f"  page {n}/{len(starts)}  ({len(page.get('store_items', []))} items, ~{(len(starts) - n) / rate / 60:.1f} min left)")
        time.sleep(delay)
    print("fetch complete")


def iter_items():
    """Yield every raw store item from the cached pages (deduplicated by appid)."""
    seen = set()
    for path in sorted(config.RAW.glob("page_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for it in json.load(f):
                appid = it.get("appid")
                if appid is None or appid in seen:
                    continue
                seen.add(appid)
                yield it


# ---------- on-demand lookups ----------

class SteamError(Exception):
    """Steam could not be reached, or answered with something unusable."""


class SteamLimit(SteamError):
    """This process has used its daily allowance of lookups (DAILY_LIMIT)."""


MIN_INTERVAL = 1.0  # seconds between two lookups, however many threads ask
CACHE_DAYS = 30
ITEMS_PER_CALL = 50
DAILY_LIMIT = None  # lookups per day; a public server sets one so visitors cannot drive unbounded traffic
_lock = threading.Lock()
_last = 0.0
_used = {"day": None, "n": 0}


def _polite_get(path: str, body: dict) -> dict:
    """One store API call at a time, spaced MIN_INTERVAL apart, with a short retry budget."""
    global _last
    with _lock:
        today = time.strftime("%Y-%m-%d")
        if _used["day"] != today:
            _used.update(day=today, n=0)
        if DAILY_LIMIT and _used["n"] >= DAILY_LIMIT:
            raise SteamLimit("IdeaPlotter has used today's Steam lookups. Try again tomorrow.")
        _used["n"] += 1
        time.sleep(max(0.0, _last + MIN_INTERVAL - time.monotonic()))
        try:
            return _get(path, {"input_json": json.dumps(body, separators=(",", ":"))}, retries=3)["response"]
        except (OSError, RuntimeError, ValueError, KeyError) as e:  # HTTP/URL errors are OSErrors
            raise SteamError(str(e)) from e
        finally:
            _last = time.monotonic()


def _cache_path(appid: int):
    return config.STEAM_CACHE / f"{int(appid)}.json"


def _cached(appid: int, max_age: float) -> dict | None:
    path = _cache_path(appid)
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec["item"] if time.time() - rec.get("fetched", 0) <= max_age else None


def get_items(appids, max_age_days: float = CACHE_DAYS) -> dict[int, dict]:
    """Raw store items (same shape as the crawl's) for these appids, from the disk cache or Steam.

    Games Steam does not know, or no longer lists, are left out of the result."""
    max_age = max_age_days * 86400
    out, todo = {}, []
    for a in dict.fromkeys(int(a) for a in appids):
        it = _cached(a, max_age)
        if it is not None:
            out[a] = it
        else:
            todo.append(a)
    if todo:
        config.STEAM_CACHE.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), ITEMS_PER_CALL):
        ids = [{"appid": a} for a in todo[i : i + ITEMS_PER_CALL]]
        body = {"ids": ids, "context": CONTEXT, "data_request": DATA_REQUEST}
        for it in _polite_get("IStoreBrowseService/GetItems/v1/", body).get("store_items", []):
            if it.get("success") != 1 or not it.get("appid"):
                continue
            a = int(it["appid"])
            _cache_path(a).write_text(json.dumps({"fetched": time.time(), "item": it}, separators=(",", ":")),
                                      encoding="utf-8")
            out[a] = it
    return out


def tag_names() -> dict[str, str]:
    """Steam's tag id -> name table (data/tags.json); a snapshot user fetches it once, on first need."""
    if not config.TAGS.exists():
        tags = _polite_get("IStoreService/GetTagList/v1/", {"language": "english"})["tags"]
        config.TAGS.parent.mkdir(parents=True, exist_ok=True)
        config.TAGS.write_text(json.dumps({str(t["tagid"]): t["name"] for t in tags}, indent=0), encoding="utf-8")
    return json.loads(config.TAGS.read_text(encoding="utf-8"))


def search(term: str, limit: int = 10) -> list[dict]:
    """Steam's own store search (the suggestions under its search box), games only."""
    body = {
        "context": CONTEXT,
        "search_term": term,
        "max_results": limit,
        "filters": {"type_filters": {"include_games": True}},
        "data_request": {"include_basic_info": True},
    }
    items = _polite_get("IStoreQueryService/SearchSuggestions/v1/", body).get("store_items", [])
    return [{"appid": int(it["appid"]), "name": it.get("name", "")} for it in items if it.get("appid")]
