"""Command line: python -m ideaplotter <command>."""
import argparse
import json
import os
import time

from . import config


def cmd_download(a):
    from . import snapshot

    snapshot.download(a.url, force=a.force)


def cmd_export(a):
    from . import snapshot

    snapshot.export()


def cmd_fetch(a):
    from . import steam

    steam.fetch(delay=a.delay, refresh=a.refresh)


def cmd_build(a):
    from . import corpus, embed, space

    df = corpus.build(min_text=a.min_text, min_latin=a.min_latin)
    embed.embed_corpus(df, a.embed_model, batch_size=a.batch_size)
    space.analyze(k_clusters=a.clusters, make_map=not a.no_map)


def cmd_map3d(a):
    import numpy as np

    from . import space

    print("map: UMAP 3-D layout for the explorer")
    E = np.load(config.SPACE / "emb.npy").astype(np.float32)
    nn_sim = np.load(config.SPACE / "nn_sim.npy").astype(np.float32)
    xyz = space.umap_layout(E, nn_sim, np.load(config.SPACE / "nn_idx.npy"), dims=3)
    np.save(config.SPACE / "xyz.npy", xyz)


def cmd_voids(a):
    from . import voids
    from .space import Space

    lo, _, hi = a.anchors.partition("-")
    voids.find(Space(), n_candidates=a.candidates, anchors=(int(lo), int(hi or lo)), top=a.top,
               min_reviews=a.min_reviews, min_coherence=a.coherence, diversity=a.diversity, steps=a.steps,
               seed=a.seed)


def _llm(a):
    from . import llm

    m = llm.make(a.backend, a.model, a.effort)
    print(f"LLM: {m.label}")
    return m


def cmd_calibrate(a):
    from . import decode
    from .embed import Embedder
    from .space import Space

    space = Space()
    decode.calibrate(space, Embedder(space.model_name), _llm(a), n=a.n)


def cmd_ideas(a):
    from . import decode, report, voids
    from .embed import Embedder
    from .space import Space

    space = Space()
    vs, X = voids.load()
    if a.limit:
        vs, X = vs[: a.limit], X[: a.limit]
    decode.run(space, Embedder(space.model_name), _llm(a), vs, X, rounds=a.rounds, workers=a.workers, fresh=a.fresh)
    report.write(space, decode.rank(report.load_results(space)))  # batch + map ideas, re-measured


def cmd_report(a):
    from . import decode, report
    from .space import Space

    space = Space()
    report.write(space, decode.rank(report.load_results(space)))


def cmd_serve(a):
    from . import server

    server.serve(a.host, a.port, a.backend, a.model, a.effort, open_browser=a.open, base_path=a.base_path,
                 daily_limit=a.daily_limit, visitor_limit=a.visitor_limit, steam_daily_limit=a.steam_daily_limit)


def cmd_run(a):
    t0 = time.time()
    if not any(config.RAW.glob("page_*.json.gz")) or a.refresh:
        cmd_fetch(a)
    cmd_build(a)
    cmd_voids(a)
    cmd_ideas(a)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(prog="ideaplotter", description="Find empty regions of the Steam game space and turn them into game ideas.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_fetch(s):
        s.add_argument("--delay", type=float, default=0.5, help="seconds between Steam API pages")
        s.add_argument("--refresh", action="store_true", help="re-download pages already on disk")

    def add_build(s):
        s.add_argument("--embed-model", default=config.EMBED_MODEL)
        s.add_argument("--batch-size", type=int, default=32)
        s.add_argument("--min-text", type=int, default=120, help="min characters of store text to keep a game")
        s.add_argument("--min-latin", type=float, default=0.7, help="min share of Latin-script letters (drops non-English pages)")
        s.add_argument("--clusters", type=int, default=200)
        s.add_argument("--no-map", action="store_true", help="skip the UMAP layout")

    def add_voids(s):
        s.add_argument("--candidates", type=int, default=4096, help="anchor blends to optimise")
        s.add_argument("--anchors", default="2-4", help="games per blend, e.g. 2-4 (hybrids) or 6-10 (deep voids)")
        s.add_argument("--top", type=int, default=40, help="voids to keep after diversity filtering")
        s.add_argument("--min-reviews", type=int, default=20, help="anchor games need at least this many reviews")
        s.add_argument("--coherence", type=float, default=0.82,
                       help="min length of the un-normalised blend (1 = one game); lower allows wilder mixes")
        s.add_argument("--diversity", type=float, default=0.88, help="max cosine between two kept voids")
        s.add_argument("--steps", type=int, default=200)
        s.add_argument("--seed", type=int, default=0)

    def add_llm(s):
        s.add_argument("--backend", choices=["auto", "api", "cli", "openrouter"],
                       default=os.environ.get("IDEAPLOTTER_BACKEND") or "auto",
                       help="api = Anthropic SDK (ANTHROPIC_API_KEY); cli = local `claude -p`; openrouter = any OpenRouter model (OPENROUTER_API_KEY)")
        s.add_argument("--model", default=config.LLM_MODEL)
        s.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])

    def add_ideas(s):
        add_llm(s)
        s.add_argument("--rounds", type=int, default=3, help="write -> embed -> measure -> revise rounds per void")
        s.add_argument("--workers", type=int, default=4)
        s.add_argument("--limit", type=int, default=0, help="only decode the first N voids")
        s.add_argument("--fresh", action="store_true", help="ignore cached ideas in out/ideas/")

    s = sub.add_parser("download", help="download the prebuilt game map (start here)")
    s.add_argument("--url", help=f"snapshot location (default: release {config.SNAPSHOT_TAG} on GitHub)")
    s.add_argument("--force", action="store_true", help="re-download files that are already present")
    s.set_defaults(fn=cmd_download)
    s = sub.add_parser("export", help="pack data/space/ into out/snapshot/ for publishing")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("fetch", help="crawl the whole Steam catalog (only to build a new snapshot)")
    add_fetch(s)
    s.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("build", help="clean, embed, and analyse the corpus")
    add_build(s)
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("map3d", help="(re)build only the 3-D map layout from an existing build")
    s.set_defaults(fn=cmd_map3d)
    s = sub.add_parser("voids", help="search for empty regions")
    add_voids(s)
    s.set_defaults(fn=cmd_voids)
    s = sub.add_parser("calibrate", help="measure the rewrite noise floor (optional)")
    add_llm(s)
    s.add_argument("--n", type=int, default=16)
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("ideas", help="decode voids into ranked game ideas")
    add_ideas(s)
    s.set_defaults(fn=cmd_ideas)
    s = sub.add_parser("report", help="rebuild out/ideas.md and out/explorer.html from cached ideas")
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("serve", help="interactive map: lasso an area, get a game idea for it")
    add_llm(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action=argparse.BooleanOptionalAction, default=False,
                   help="open the explorer in your browser once the server is up")
    s.add_argument("--base-path", default="", help="serve under a path prefix, e.g. /ideaplotter behind a shared domain")
    s.add_argument("--daily-limit", type=int, help="public server: ideas per day for everyone together (one at a time)")
    s.add_argument("--visitor-limit", type=int, help="public server: ideas per day per visitor")
    s.add_argument("--steam-daily-limit", type=int, help="public server: Steam lookups per day (search, add, store text)")
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("run", help="fetch (if needed) + build + voids + ideas")
    add_fetch(s)
    add_build(s)
    add_voids(s)
    add_ideas(s)
    s.set_defaults(fn=cmd_run)

    a = p.parse_args()
    a.fn(a)
