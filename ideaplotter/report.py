"""Write ranked ideas as Markdown (out/ideas.md) and as an interactive map (out/explorer.html).

The same page template also powers the live explorer served by `ideaplotter serve`;
there the map data is fetched from the server instead of being embedded.
"""
import base64
import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import config
from .space import Space

TEMPLATE = Path(__file__).with_name("explorer_template.html")


def _pct(x: float) -> str:
    return f"{x:.1%}" if x < 0.995 else f"{x:.2%}"


def _rewrite_sim():
    cal = config.SPACE / "calibration.json"
    return json.loads(cal.read_text())["median_self_sim"] if cal.exists() else None


def load_results(space: Space | None = None) -> list[dict]:
    """Every cached idea. With a space, novelty and neighbours are re-measured against the current corpus,
    so ideas stay honest after the Steam catalog is re-crawled."""
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((config.OUT / "ideas").glob("*.json"))]
    if space is not None and results:
        V = np.array([r["vec"] for r in results], dtype=np.float32)
        sims, idx = space.neighbors(V, 10)
        for r, s, ix in zip(results, sims, idx):
            r["nearest_sim"] = float(s[0])
            r["novelty"] = 1 - float(s[0])
            r["isolation_pct"] = float(space.isolation_percentile(float(s[0])))
            r["neighbors"] = [dict(space.game_card(j), sim=round(float(v), 3)) for v, j in zip(s, ix)]
    return results


def markdown(space: Space, ranked: list[dict]) -> str:
    typical = float(np.median(space.isolation_ref))
    rewrite = _rewrite_sim()
    out = [
        "# IdeaPlotter: game ideas from the empty regions of Steam",
        "",
        f"Generated {dt.date.today().isoformat()} from {len(space):,} Steam games embedded with `{space.model_name}`.",
        "",
        "**Novelty** is 1 minus the cosine similarity between the idea's own store page and the closest "
        f"real Steam game, measured in the same embedding space. The typical Steam game's closest neighbour is at "
        f"{typical:.3f} similarity (novelty {1 - typical:.3f})."
        + (f" An LLM rewrite of an existing game lands at {rewrite:.3f} from the original, so anything with a "
           f"nearest similarity well below that is further from every game than a paraphrase is from its source."
           if rewrite else ""),
        "**Isolation** is the share of Steam games whose own closest neighbour is nearer than the idea's.",
        "**Landed** means the idea's page sits closer to the void it was aimed at than to any existing game.",
        "",
        "| # | Idea | Novelty | Isolation | Closest existing game | Landed | Viability |",
        "|---|------|---------|-----------|-----------------------|--------|-----------|",
    ]
    for r in ranked:
        n0 = r["neighbors"][0]
        out.append(f"| {r['rank']} | [{r['idea']['title']}](#{r['rank']}) | {r['novelty']:.3f} | {_pct(r['isolation_pct'])} "
                   f"| {n0['name']} ({n0['sim']:.2f}) | {'yes' if r['landed'] else 'no'} | {r['idea']['viability']}/10 |")
    for r in ranked:
        i = r["idea"]
        out += [
            "", f'<a id="{r["rank"]}"></a>', f"## {r['rank']}. {i['title']}", "", f"> {i['short_description']}", "",
            f"**Novelty {r['novelty']:.3f}** (more isolated than {_pct(r['isolation_pct'])} of Steam games) · "
            f"fit to its void {r['target_sim']:.3f} · {'landed in the void' if r['landed'] else 'did not land'} · "
            f"viability {i['viability']}/10 ({i['viability_note']})",
            "", i["about"], "",
            f"**Tags:** {', '.join(i['tags'])}", "",
            f"**Core loop:** {i['core_loop']}", "",
            f"**Why it's new:** {i['why_new']}", "",
            "**Recipe (the void is a blend of):** "
            + ", ".join(f"{a['weight']:.0%} [{a['name']}]({a['url']})" for a in r["void"]["anchors"]), "",
            "**Closest existing games:** "
            + ", ".join(f"[{n['name']}]({n['url']}) ({n['sim']:.2f})" for n in r["neighbors"][:5]), "",
            "**Rounds:** " + " → ".join(f"{x['title']} (target {x['target_sim']:.2f}, nearest {x['nearest_sim']:.2f})"
                                        for x in r["rounds"]),
        ]
    return "\n".join(out) + "\n"


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def map_payload(space: Space, holes: tuple | None = None) -> dict:
    """Everything the browser needs to draw the cloud: positions, clusters, popularity, names.
    holes = (3-D, 2-D) holes from an earlier payload, to skip finding them again."""
    if space.xy is None or space.xyz is None:
        raise RuntimeError("no map; rerun `build` without --no-map, or run `python -m ideaplotter map3d` for the 3-D one")
    games = space.games
    reviews = np.clip(np.log10(games["review_count"].fillna(0).to_numpy() + 1) * 48, 0, 255).astype(np.uint8)
    k = len(space.cluster_names)
    sizes = np.bincount(space.cluster, minlength=k)
    center = lambda P, c: np.median(P[space.cluster == c], axis=0).round(1).tolist() if sizes[c] else [0] * P.shape[1]
    lo = min(0.5, np.floor(np.quantile(space.isolation_ref, 0.002) * 20) / 20)
    hist, edges = np.histogram(space.isolation_ref, bins=56, range=(lo, 1.0))
    # default view frames the core of the cloud; small outlying islands stay reachable by panning
    def frame(P):
        v0, v1 = np.quantile(P, 0.001, axis=0), np.quantile(P, 0.999, axis=0)
        pad = (v1 - v0) * 0.03
        return [float(v) for v in np.concatenate([v0 - pad, v1 + pad])]
    view, view2 = frame(space.xyz_q), frame(space.xy_q)
    from .holes import find_holes, find_holes_2d
    return {
        "meta": {
            "n": len(games), "date": dt.date.today().isoformat(), "model": space.model_name,
            "dim": int(space.E.shape[1]),
            "typical_nn": float(np.median(space.isolation_ref)), "rewrite_sim": _rewrite_sim(),
            "hist": hist.tolist(), "hist_lo": float(edges[0]), "hist_hi": float(edges[-1]),
            "view": view, "view2": view2,
        },
        "holes": holes[0] if holes else find_holes(space, view),
        "holes2": holes[1] if holes else find_holes_2d(space, view2),
        "xyz": _b64(np.round(space.xyz_q).astype(np.uint16)),
        "xy": _b64(np.round(space.xy_q).astype(np.uint16)),
        "cluster": _b64(space.cluster.astype(np.uint8)),
        "rev": _b64(reviews),
        "appid": _b64(games["appid"].to_numpy().astype(np.uint32)),
        "names": games["name"].tolist(),
        "clusters": [{"name": n, "xyz": center(space.xyz_q, c), "xy": center(space.xy_q, c), "size": int(sizes[c])}
                     for c, n in enumerate(space.cluster_names)],
    }


def ui_idea(space: Space, r: dict, appid_to_row: dict) -> dict:
    vec = np.asarray(r["vec"], dtype=np.float32)
    ref = lambda c: {"row": appid_to_row.get(c["appid"]), "name": c["name"], "appid": c["appid"]}
    return {
        "id": r["void_id"], "source": r.get("source", "global"), "created": r.get("created"),
        **r["idea"],
        "novelty": r["novelty"], "nearest_sim": r["nearest_sim"], "isolation_pct": r["isolation_pct"],
        "target_sim": r["target_sim"], "landed": r["landed"], "rounds": r["rounds"], "best_round": r["best_round"],
        "void_novelty": r["void"]["novelty"], "predicted_tags": r["void"]["predicted_tags"][:10],
        # hole ideas are pinned where they were aimed; others (and holes from the old 2-D map) sit among their nearest games
        "xyz": ((r.get("hole") or {}).get("xyz") or space.place_3d(vec)[0]) if space.xyz is not None else None,
        "xy": (r.get("hole") or {}).get("xy") or space.to_q(space.place_2d(vec)[0]) if space.xy is not None else None,
        "area": bool(r.get("selection")),
        "selection_count": (r.get("selection") or {}).get("count"),
        "hole": r.get("hole"),
        "blend_rows": r.get("blend_rows"),
        "surrounding_regions": r["void"].get("surrounding_regions"),
        "neighbors": [ref(n) | {"sim": n["sim"]} for n in r["neighbors"][:6]],
        "anchors": [ref(a) | {"weight": a["weight"]} for a in r["void"]["anchors"]],
    }


def ui_ideas(space: Space, results: list[dict]) -> list[dict]:
    appid_to_row = {int(a): i for i, a in enumerate(space.games["appid"])}
    return [ui_idea(space, r, appid_to_row) for r in sorted(results, key=lambda r: -r["novelty"])]


def render_page(data: dict | None) -> str:
    """The explorer body. With data=None the page fetches /api/map and /api/ideas (live server)."""
    blob = "null" if data is None else json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", blob)


def full_document(body: str) -> str:
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            "</head><body>\n" + body + "\n</body></html>\n")


def write(space: Space, ranked: list[dict]) -> None:
    config.OUT.mkdir(parents=True, exist_ok=True)
    (config.OUT / "ideas.md").write_text(markdown(space, ranked), encoding="utf-8")
    (config.OUT / "ideas.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "vec"} for r in ranked], indent=1, ensure_ascii=False),
        encoding="utf-8")
    if space.xyz is not None:
        data = map_payload(space)
        data["ideas"] = ui_ideas(space, ranked)
        body = render_page(data)
        (config.OUT / "explorer_body.html").write_text(body, encoding="utf-8")  # page content only, for publishing
        (config.OUT / "explorer.html").write_text(full_document(body), encoding="utf-8")
    print(f"report: {len(ranked)} ideas -> out/ideas.md, out/ideas.json, out/explorer.html")
