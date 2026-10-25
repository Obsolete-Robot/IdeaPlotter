"""Local web app: browse the cloud of Steam games and turn empty places into game ideas.

    python -m ideaplotter serve            -> http://127.0.0.1:8765   (or double-click run.bat)

GET  /                 the explorer page
GET  /api/map          point cloud payload (positions, clusters, names, holes)
GET  /api/ideas        every idea generated so far
GET  /api/game/<row>   one game: store card, closest games, furthest games
GET  /api/store/<appid> its About text, looked up on Steam (cached on disk)
GET  /api/search?q=    Steam's own store search, for games newer than the snapshot
POST /api/add          {appid}               -> {row}: fetch one game from Steam, embed it, put it on the map
POST /api/select       {rows}                -> summary of the lassoed games
POST /api/explore      {mode, rounds, ...}   -> {job}: find a void and write a game for it
                         mode "area"   {rows | idea, anchors}  emptiest blend of the lassoed games (or an earlier idea's)
                         mode "blend"  {rows}                  emptiest mix of games you picked
                         mode "hole"   {center, radius}        a gap in the 3-D map, triangulated from its rim
                         mode "global" {}                   the emptiest blend anywhere on Steam
GET  /api/job/<id>     progress events and, when done, the idea
"""
import datetime as dt
import hashlib
import json
import secrets
import threading
import traceback
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import config, corpus, decode, holes, llm, report, steam, voids
from .embed import Embedder
from .progress import Status, dir_size, fmt_bytes
from .space import Space


class Quota:
    """Limits for a public server: one idea at a time, at most `daily` ideas a day in total and
    `per_visitor` per visitor. Counts survive restarts (out/quota.json); visitors are stored hashed."""

    def __init__(self, daily: int | None, per_visitor: int | None):
        self.daily, self.per_visitor = daily, per_visitor
        self.path = config.OUT / "quota.json"
        self._lock = threading.Lock()
        try:
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.state = {}

    @property
    def active(self) -> bool:
        return bool(self.daily or self.per_visitor)

    def admit(self, visitor: str, busy: bool) -> None:
        """Count one idea for this visitor, or raise ValueError with a message for the page."""
        if not self.active:
            return
        with self._lock:
            today = dt.date.today().isoformat()
            if self.state.get("day") != today:
                self.state = {"day": today, "total": 0, "visitors": {}}
            if busy:
                raise ValueError("Another visitor's game is being written right now. Try again in a minute.")
            if self.daily and self.state["total"] >= self.daily:
                raise ValueError(f"All {self.daily} of today's ideas have been written. Come back tomorrow!")
            key = hashlib.sha256(visitor.encode()).hexdigest()[:16]
            mine = self.state["visitors"].get(key, 0)
            if self.per_visitor and mine >= self.per_visitor:
                raise ValueError(f"You have written your {self.per_visitor} ideas for today. Come back tomorrow!")
            self.state["total"] += 1
            self.state["visitors"][key] = mine + 1
            config.OUT.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.state), encoding="utf-8")


def load_embedder(model_name: str) -> Embedder:
    """The embedding model, with a live line while it downloads (first start only) and loads."""
    from huggingface_hub import constants, try_to_load_from_cache
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    def cached(name: str) -> bool:
        return isinstance(try_to_load_from_cache(model_name, name), str)

    loading = f"loading the embedding model {model_name}"
    if cached("model.safetensors"):
        with Status(loading):
            return Embedder(model_name)
    home, base = constants.HF_HOME, dir_size(constants.HF_HOME)

    def text() -> str:
        if cached("model.safetensors"):
            return loading
        return f"downloading the embedding model (first start): {fmt_bytes(max(0, dir_size(home) - base))} of ~1.2 GB"

    disable_progress_bars()  # one line of our own instead of a bar per file
    try:
        with Status(text):
            return Embedder(model_name)
    finally:
        enable_progress_bars()


class App:
    def __init__(self, backend: str, model: str, effort: str, quota: Quota | None = None):
        self.quota = quota or Quota(None, None)
        with Status("loading the game map"):
            self.space = Space()
        if self.space.xy is None or self.space.xyz is None:
            raise SystemExit("map layout missing; run `python -m ideaplotter download` for the prebuilt map")
        print(f"  {len(self.space):,} games")
        self.embedder = load_embedder(self.space.model_name)
        self.llm_args = (backend, model, effort)
        self._llm = None
        self._llm_lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self._add_lock = threading.Lock()
        self.holes = None
        with Status("finding the empty spots and preparing the map for the browser"):
            self._refresh_map()
        self.explored = [np.asarray(r["vec"], dtype=np.float32) for r in report.load_results(self.space)]

    def _refresh_map(self):
        payload = report.map_payload(self.space, self.holes)
        self.holes = (payload["holes"], payload["holes2"])  # found once; a few added games do not move them
        self.map_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self._appid_row = {int(a): i for i, a in enumerate(self.space.games["appid"])}

    def get_llm(self):
        with self._llm_lock:
            if self._llm is None:
                self._llm = llm.make(*self.llm_args)
                print(f"LLM: {self._llm.label}")
            return self._llm

    def ideas(self) -> list[dict]:
        return report.ui_ideas(self.space, report.load_results(self.space))

    def area_rows(self, spec: dict) -> np.ndarray:
        """The games of a lassoed area: sent as rows, or taken from the idea it was first explored for."""
        if spec.get("idea"):
            path = config.OUT / "ideas" / f"{spec['idea']}.json"
            if path.parent != (config.OUT / "ideas") or not path.exists():
                raise ValueError("That idea is no longer on disk.")
            sel = json.loads(path.read_text(encoding="utf-8")).get("selection") or {}
            if "rows" in sel:
                return np.asarray(sel["rows"], dtype=np.int64)
            return self.space.rows_in_polygon(sel["polygon"])  # saved by the 2-D map
        rows = np.unique(np.asarray(spec.get("rows") or [], dtype=np.int64))
        return rows[(rows >= 0) & (rows < len(self.space))]

    def select(self, spec: dict) -> dict:
        rows = self.area_rows(spec)
        if not len(rows):
            return {"count": 0}
        return self.space.summarize(rows)

    def game(self, row: int) -> dict:
        card = self.space.game_card(row)
        card.update(self.space.closest_and_furthest(row))
        card["row"] = int(row)
        card["nearest_twin_sim"] = float(self.space.nn_sim[row, 0])
        card["isolation_pct"] = float(self.space.isolation_percentile(card["nearest_twin_sim"]))
        card["region"] = self.space.cluster_names[self.space.cluster[row]]
        return card

    def store_text(self, appid: int) -> dict:
        row = self._appid_row.get(appid)
        if row is not None and "about" in self.space.games:  # a full `build` table has the text: no Steam call
            g = self.space.games.iloc[row]
            if isinstance(g["about"], str):
                return {"appid": appid, "short_description": g["short_description"], "about": g["about"][:1500]}
        it = steam.get_items([appid]).get(appid)
        if it is None:
            raise LookupError("Steam no longer lists this game.")
        return {"appid": appid,
                "short_description": corpus.clean_bbcode((it.get("basic_info") or {}).get("short_description", "")),
                "about": corpus.clean_bbcode(it.get("full_description_bbcode", ""))[:1500]}

    def search(self, term: str) -> list[dict]:
        return [dict(r, row=self._appid_row.get(r["appid"])) for r in steam.search(term)]

    def add(self, appid: int) -> dict:
        """Look one game up on Steam, embed it like every other game, and put it on the map."""
        with self._add_lock:
            if appid in self._appid_row:
                return {"row": self._appid_row[appid], "added": False}
            if self.busy():
                raise ValueError("Wait until the game being written is finished, then add this one.")
            it = steam.get_items([appid]).get(appid)
            if it is None:
                raise LookupError("Steam has no store page for that game.")
            g = corpus.parse_item(it, steam.tag_names())
            if len(g["short_description"]) + len(g["about"]) < 50:
                raise ValueError(f"{g['name']} has almost no store text yet, so there is nothing to place it by.")
            vec = self.embedder.encode([corpus.doc_text(g["name"], g["short_description"], g["about"], g["tags"])])[0]
            row = self.space.add_game({
                "appid": appid, "name": g["name"], "short_description": g["short_description"], "tags": g["tags"],
                "review_count": g["review_count"], "nsfw": bool(config.NSFW_TAGS.intersection(g["tags"][:10])),
                "text_len": len(g["short_description"]) + len(g["about"]),
            }, vec)
            self._refresh_map()
            print(f"  added {g['name']} ({appid}) as game {row:,}")
            return {"row": row, "added": True}

    def busy(self) -> bool:
        return any(j["status"] in ("searching", "writing") for j in self.jobs.values())

    def start_job(self, spec: dict, visitor: str = "local") -> str:
        self.quota.admit(visitor, self.quota.active and self.busy())
        for old in [k for k, j in self.jobs.items() if j["status"] in ("done", "error")][:-50]:
            del self.jobs[old]  # finished jobs are only polled briefly; keep the last few
        job_id = secrets.token_hex(4)
        self.jobs[job_id] = {"id": job_id, "mode": spec["mode"], "status": "searching", "events": [],
                             "idea": None, "error": None}
        threading.Thread(target=self._run_job, args=(job_id, spec), daemon=True).start()
        return job_id

    def _event(self, job, kind, **data):
        job["events"].append({"kind": kind, "t": dt.datetime.now().isoformat(timespec="seconds"), **data})

    def _find_void(self, job, spec):
        """Turn a request into (void, target vector, fields stored with the idea)."""
        exclude = np.array(self.explored) if self.explored else None
        mode = spec["mode"]
        if mode == "area":
            rows = self.area_rows(spec)
            if len(rows) < 2:
                raise ValueError("Draw around at least two games.")
            self._event(job, "search", count=int(len(rows)))
            lo, _, hi = str(spec.get("anchors", "2-4")).partition("-")
            void, x = voids.find_in_region(self.space, rows, anchors=(int(lo), int(hi or lo)), exclude=exclude)
            extra = {"selection": {"rows": rows.tolist(), "count": int(len(rows))}}
        elif mode == "blend":
            rows = [int(r) for r in spec["rows"]]
            self._event(job, "search", count=len(rows))
            void, x = voids.find_for_blend(self.space, rows, exclude=exclude)
            extra = {"blend_rows": rows}
        elif mode == "hole":
            center, radius = [float(v) for v in spec["center"]], float(spec.get("radius") or 600)
            self._event(job, "search", count=0)
            void, x = holes.hole_target(self.space, center, radius, exclude=exclude)
            extra = {"hole": {"xyz" if len(center) == 3 else "xy": [round(c, 1) for c in center], "r": round(radius, 1)}}
        elif mode == "global":
            self._event(job, "search", count=len(self.space))
            void, x = voids.find_global_one(self.space, exclude=exclude)
            extra = {}
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return void, x, extra

    def _run_job(self, job_id, spec):
        job = self.jobs[job_id]
        try:
            void, x, extra = self._find_void(job, spec)
            void["id"] = f"{spec['mode']}-{dt.datetime.now():%Y%m%d-%H%M%S}-{job_id}"
            job["void"] = {
                "novelty": void["novelty"], "nearest_sim": void["nearest_sim"], "isolation_pct": void["isolation_pct"],
                "pool_size": void.get("pool_size"), "candidates": void.get("candidates"),
                "coherence": void["coherence"],
                "xyz": extra.get("hole", {}).get("xyz") or self.space.place_3d(x)[0],
                "xy": extra.get("hole", {}).get("xy") or self.space.to_q(self.space.place_2d(x)[0]),
                "anchors": [{"name": a["name"], "appid": a["appid"], "weight": a["weight"],
                             "row": self._appid_row.get(a["appid"])} for a in void["anchors"]],
                "predicted_tags": void["predicted_tags"][:10],
                "nearest": [{"name": n["name"], "sim": n["sim"]} for n in void["neighbors"][:3]],
                "surrounding_regions": void.get("surrounding_regions"),
            }
            self._event(job, "void")
            job["status"] = "writing"
            writer = self.get_llm()
            cost0 = getattr(writer, "cost", None)  # OpenRouter reports what each call cost
            dec = decode.Decoder(self.space, self.embedder, writer, rounds=spec["rounds"])
            r = dec.decode(void, x, on_round=lambda info: self._event(job, "round", **info))
            print(f"  idea {void['id']}: {r['idea']['title']!r}, {len(r['rounds'])} round(s)"
                  + (f", ${writer.cost - cost0:.4f}" if cost0 is not None else ""))
            r["vec"] = r.pop("_vec").round(5).tolist()
            r["source"] = spec["mode"]
            r["created"] = dt.datetime.now().isoformat(timespec="seconds")
            r.update(extra)
            (config.OUT / "ideas").mkdir(parents=True, exist_ok=True)
            (config.OUT / "ideas" / f"{void['id']}.json").write_text(json.dumps(r, ensure_ascii=False),
                                                                     encoding="utf-8")
            self.explored.append(x.astype(np.float32))
            job["idea"] = report.ui_idea(self.space, r, self._appid_row)
            job["status"] = "done"
            self._event(job, "done")
        except Exception as e:  # report every failure to the page instead of dying silently
            traceback.print_exc()
            job["status"], job["error"] = "error", str(e) or e.__class__.__name__


MAX_BODY = 8 << 20  # a lasso around the whole map sends ~1.5 MB of rows


def make_handler(app: App, base_path: str = ""):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if "/api/job/" not in self.path:
                print("  " + fmt % args)

        def _rebase(self) -> bool:
            """Serve under base_path (e.g. /ideaplotter behind a shared domain): strip it from self.path.
            The page uses relative URLs, so the bare base path is redirected to base_path + "/"."""
            if not base_path:
                return True
            path, sep, query = self.path.partition("?")
            if path == base_path:
                self.send_response(301)
                self.send_header("Location", base_path + "/" + sep + query)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False
            if path.startswith(base_path + "/"):
                self.path = self.path[len(base_path):]
            return True

        def _visitor(self) -> str:
            # behind Cloudflare every request comes from the tunnel; the visitor's address is in this header
            return self.headers.get("CF-Connecting-IP") or self.client_address[0]

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def _steam(self, fn, *args):
            """Answer with a lookup that needs Steam; say so plainly when Steam cannot be reached."""
            try:
                return self._json(fn(*args))
            except LookupError as e:
                return self._json({"error": str(e)}, 404)
            except steam.SteamLimit as e:
                return self._json({"error": str(e)}, 429)
            except steam.SteamError as e:
                print(f"  Steam lookup failed: {e}")
                return self._json({"error": "Couldn't reach Steam. Check your connection and try again."}, 502)

        def do_GET(self):
            if not self._rebase():
                return
            url = urllib.parse.urlsplit(self.path)
            if url.path.startswith("/api/store/"):
                try:
                    appid = int(url.path.rsplit("/", 1)[-1])
                except ValueError:
                    return self._json({"error": "bad app id"}, 400)
                return self._steam(app.store_text, appid)
            if url.path == "/api/search":
                term = (urllib.parse.parse_qs(url.query).get("q") or [""])[0].strip()
                if not 2 <= len(term) <= 100:
                    return self._json({"error": "Type at least two letters."}, 400)
                return self._steam(app.search, term)
            if self.path in ("/", "/index.html"):
                page = report.full_document(report.render_page(None)).encode()  # re-read so template edits show on reload
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/map":
                return self._send(200, app.map_json)
            if self.path == "/api/ideas":
                return self._json(app.ideas())
            if self.path.startswith("/api/game/"):
                try:
                    row = int(self.path.rsplit("/", 1)[-1])
                except ValueError:
                    return self._json({"error": "bad game id"}, 400)
                if not 0 <= row < len(app.space):
                    return self._json({"error": "unknown game"}, 404)
                return self._json(app.game(row))
            if self.path.startswith("/api/job/"):
                job = app.jobs.get(self.path.rsplit("/", 1)[-1])
                return self._json(job) if job else self._json({"error": "unknown job"}, 404)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._rebase():
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if size > MAX_BODY:
                    return self._json({"error": "Request too large."}, 413)
                body = json.loads(self.rfile.read(size) or b"{}")
                if self.path == "/api/add":
                    return self._steam(app.add, int(body["appid"]))
                mode = body.get("mode", "area")
                if (self.path == "/api/select" or mode == "area") and not (body.get("rows") or body.get("idea")):
                    return self._json({"error": "Draw around some games first."}, 400)
                if self.path == "/api/select":
                    return self._json(app.select(body))
                if self.path == "/api/explore":
                    if mode == "blend" and not 2 <= len(body.get("rows") or []) <= 8:
                        return self._json({"error": "Pick between 2 and 8 games to combine."}, 400)
                    if mode == "hole" and len(body.get("center") or []) not in (2, 3):
                        return self._json({"error": "Missing the spot to explore."}, 400)
                    body["mode"] = mode
                    body["rounds"] = max(1, min(4, int(body.get("rounds", 2))))
                    return self._json({"job": app.start_job(body, self._visitor())})
                self._json({"error": "not found"}, 404)
            except (ValueError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def _already_running(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/ideas", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def serve(host: str = "127.0.0.1", port: int = 8765, backend: str = "auto", model: str = config.LLM_MODEL,
          effort: str = "high", open_browser: bool = False, base_path: str = "", daily_limit: int | None = None,
          visitor_limit: int | None = None, steam_daily_limit: int | None = None) -> None:
    base_path = "/" + base_path.strip("/") if base_path.strip("/") else ""
    url = f"http://{host}:{port}{base_path}/"
    if _already_running(host, port):
        print(f"IdeaPlotter is already running at {url}")
        if open_browser:
            webbrowser.open(url)
        return
    steam.DAILY_LIMIT = steam_daily_limit
    quota = Quota(daily_limit, visitor_limit)
    if quota.active or steam_daily_limit:
        print(f"public limits: {daily_limit or 'unlimited'} ideas/day, {visitor_limit or 'unlimited'} per visitor, "
              f"one at a time; {steam_daily_limit or 'unlimited'} Steam lookups/day")
    app = App(backend, model, effort, quota)
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(app, base_path))
    except OSError as e:
        raise SystemExit(f"Port {port} is in use by another program ({e}). Try: --port {port + 1}") from e
    print(f"IdeaPlotter explorer running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
