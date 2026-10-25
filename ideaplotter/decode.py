"""Decode void targets back into game concepts, with closed-loop verification.

Embeddings cannot be inverted directly, so an LLM writes a candidate game from a brief
of the void (its blend recipe, the tags a linear probe reads off the point, and the
closest existing games). The draft is then embedded with the *same* model and document
format as the Steam corpus and measured:

    target_sim   cos(draft, void point)        did it land where we aimed?
    nearest_sim  max cos(draft, any Steam game) how close is the closest real game?
    landed       target_sim > nearest_sim       the draft is nearer the void than any game

The measurements go back to the LLM for another round, and the best round is kept.
Ideas are finally ranked by novelty = 1 - nearest_sim of the decoded text itself.
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import config, steam
from .corpus import clean_bbcode, doc_text
from .embed import Embedder
from .llm import LLMError
from .space import Space

IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Store title of the game"},
        "short_description": {"type": "string", "description": "Steam capsule text, 1-2 sentences, under 300 characters"},
        "about": {"type": "string", "description": "Steam 'About This Game' text, 120-220 words, concrete mechanics"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "8-15 Steam user tags, most relevant first"},
        "core_loop": {"type": "string", "description": "What the player does minute to minute, 1-3 sentences"},
        "why_new": {"type": "string", "description": "What this does that the listed neighbouring games do not, 1-3 sentences"},
        "viability": {"type": "integer", "description": "1-10: how buildable and fun this is as a real product"},
        "viability_note": {"type": "string", "description": "One sentence justifying the viability score"},
    },
    "required": ["title", "short_description", "about", "tags", "core_loop", "why_new", "viability", "viability_note"],
    "additionalProperties": False,
}

SYSTEM = """You are a veteran game designer and a Steam store copywriter. You are given a location in a semantic map of every game on Steam: a point that no existing game occupies, described by the games and tags around it. Your job is to invent the one game that would live at that point.

What a good answer looks like:
- One coherent, playable concept with a clear core loop that a small team could build. Treat the recipe games as ingredients to fuse into a single design, not a list of features to bolt together.
- Clearly different from every listed neighbouring game. If a player would describe it as "basically <neighbour> with X", it is not far enough away.
- Written like a real Steam page, because it is measured against real Steam pages: a plain, concrete short description and an About section that says what the player actually does. Avoid hype words ("revolutionary", "unique", "like never before"). Never mention maps, embeddings, voids, or the reference games by name.
- Tags are real Steam user tags, most relevant first.
- The viability score is honest: a strange idea that would not be fun to play should score low."""


def _game_line(c: dict, extra: str = "") -> str:
    short = c.get("short_description") or ""
    return f"**{c['name']}**{extra}" + (f" — {short[:280]}" if short else "") + f"\n  Tags: {', '.join(c['tags'][:10])}"


def store_texts(appids) -> dict[int, tuple[str, str]]:
    """(short description, About text) of a few games, looked up on Steam (one batched, cached request)."""
    out = {}
    for a, it in steam.get_items(appids).items():
        short = clean_bbcode((it.get("basic_info") or {}).get("short_description", ""))
        out[a] = (short, clean_bbcode(it.get("full_description_bbcode", "")))
    return out


def fill_short(cards: list[dict]) -> None:
    """Snapshots ship without store copy, so look up the short descriptions a brief quotes (about a dozen
    games). If Steam cannot be reached, the brief describes those games by title and tags alone."""
    need = [c["appid"] for c in cards if not c.get("short_description")]
    if not need:
        return
    try:
        texts = store_texts(need)
    except steam.SteamError as e:
        print(f"  could not look up store descriptions ({e}); briefing with titles and tags only")
        return
    for c in cards:
        if not c.get("short_description") and c["appid"] in texts:
            c["short_description"] = texts[c["appid"]][0]


def brief(void: dict, typical_nn: float, rewrite_sim: float | None) -> str:
    lines = ["# The void", ""]
    lines.append(
        f"No existing Steam game sits at this point: the closest one has cosine similarity {void['nearest_sim']:.2f} "
        f"to it. Similarities are measured between store pages (title, short description, tags, About text); "
        f"for scale, a typical Steam game's closest neighbour is at {typical_nn:.2f}."
    )
    if rewrite_sim:
        lines.append(f"A rewrite of an existing game in different words typically lands at {rewrite_sim:.2f} from the original.")
    if void.get("context"):
        lines += ["", "## How this point was chosen", void["context"]]
    lines += ["", "## Blend recipe", "The point is a weighted blend of these games' positions:"]
    for a in void["anchors"]:
        lines.append(f"- {a['weight']:.0%}  " + _game_line(a))
    lines += ["", "## Tags a probe reads off this point (most characteristic first, with probability)"]
    lines.append(", ".join(f"{t} ({p:.2f})" for t, p in void["predicted_tags"]))
    if void["more_than_neighbors"] or void["less_than_neighbors"]:
        lines += ["", "## Compared with its nearest games, this point leans"]
        if void["more_than_neighbors"]:
            lines.append("- more toward: " + ", ".join(void["more_than_neighbors"]))
        if void["less_than_neighbors"]:
            lines.append("- less toward: " + ", ".join(void["less_than_neighbors"]))
    lines += ["", "## Closest existing games (the concept must be clearly different from each)"]
    for n in void["neighbors"][:6]:
        lines.append(f"- ({n['sim']:.2f}) " + _game_line(n))
    return "\n".join(lines)


def _attempt_report(i: int, a: dict) -> str:
    near = "; ".join(f"{n['name']} ({n['sim']:.2f})" for n in a["neighbors"][:4])
    verdict = ("It landed inside the void (closer to the target than to any existing game)."
               if a["landed"] else "It did not land: an existing game is closer to it than the target is.")
    return (f"### Attempt {i}: {a['idea']['title']}\n"
            f"{a['idea']['short_description']}\n"
            f"- similarity to the target point: {a['target_sim']:.3f}\n"
            f"- nearest existing games: {near}\n"
            f"- {verdict}")


def build_prompt(void_brief: str, attempts: list[dict]) -> str:
    if not attempts:
        return void_brief + "\n\n# Task\nInvent the game that lives at this point. Return it as JSON."
    hist = "\n\n".join(_attempt_report(i + 1, a) for i, a in enumerate(attempts))
    return (void_brief
            + "\n\n# Previous attempts\nEach attempt was written as a store page, embedded with the same model "
              "as every Steam game, and measured.\n\n" + hist
            + "\n\n# Task\nWrite an improved attempt. Aim to raise similarity to the target point and lower "
              "similarity to the nearest existing games: look at what the closest games share with the last "
              "draft and change that, and lean on the recipe and tag signals those games lack. Keep what "
              "worked; start over if nothing did. Return the full concept as JSON.")


class Decoder:
    def __init__(self, space: Space, embedder: Embedder, llm, rounds: int = 3):
        self.space, self.embedder, self.llm, self.rounds = space, embedder, llm, rounds
        self.tag_lookup = {t.lower(): t for t in space.all_tags}
        self.typical_nn = float(np.median(space.isolation_ref))
        cal = config.SPACE / "calibration.json"
        self.rewrite_sim = json.loads(cal.read_text())["median_self_sim"] if cal.exists() else None

    def clean_tags(self, tags: list[str]) -> list[str]:
        out = []
        for t in tags:
            canon = self.tag_lookup.get(t.strip().lower())
            if canon and canon not in out:
                out.append(canon)
        return out[:20]

    def measure(self, idea: dict, target: np.ndarray | None) -> dict:
        idea["tags"] = self.clean_tags(idea["tags"])
        v = self.embedder.encode([doc_text(idea["title"], idea["short_description"], idea["about"], idea["tags"])])[0]
        sims, idx = self.space.neighbors(v, 10)
        nearest = float(sims[0, 0])
        rec = {
            "idea": idea,
            "nearest_sim": nearest,
            "novelty": 1 - nearest,
            "isolation_pct": float(self.space.isolation_percentile(nearest)),
            "neighbors": [dict(self.space.game_card(j), sim=round(float(s), 3)) for s, j in zip(sims[0], idx[0])],
            "vec": v,
        }
        if target is not None:
            rec["target_sim"] = float(v @ target)
            rec["landed"] = rec["target_sim"] > nearest
        return rec

    def decode(self, void: dict, target: np.ndarray, on_round=None) -> dict:
        fill_short(void["anchors"] + void["neighbors"][:6])
        b = brief(void, self.typical_nn, self.rewrite_sim)
        attempts = []
        for _ in range(self.rounds):
            try:
                idea = self.llm.complete_json(SYSTEM, build_prompt(b, attempts), IDEA_SCHEMA)
            except (LLMError, json.JSONDecodeError) as e:
                if not attempts:
                    raise
                print(f"  revision round failed ({e}); keeping {len(attempts)} draft(s)")
                break
            attempts.append(self.measure(idea, target))
            if on_round:
                a = attempts[-1]
                on_round({"round": len(attempts), "title": idea["title"], "short_description": idea["short_description"],
                          "target_sim": a["target_sim"], "nearest_sim": a["nearest_sim"], "landed": a["landed"],
                          "nearest": a["neighbors"][0]["name"]})
        # Prefer drafts that landed in the void; among those (or all, if none did) take the most novel.
        best = max(attempts, key=lambda a: (a["landed"], a["novelty"]))
        return {
            "void_id": void["id"],
            "void": {k: void.get(k) for k in ("nearest_sim", "novelty", "isolation_pct", "coherence", "anchors",
                                              "predicted_tags", "context", "surrounding_regions")},
            **{k: v for k, v in best.items() if k != "vec"},
            "best_round": attempts.index(best) + 1,
            "rounds": [{k: a[k] for k in ("target_sim", "nearest_sim", "landed")} | {"title": a["idea"]["title"]}
                       for a in attempts],
            "_vec": best["vec"],
        }


def run(space: Space, embedder: Embedder, llm, voids: list[dict], X: np.ndarray, rounds: int = 3,
        workers: int = 4, fresh: bool = False) -> list[dict]:
    """Decode every void (in parallel), caching each finished idea under out/ideas/."""
    cache = config.OUT / "ideas"
    cache.mkdir(parents=True, exist_ok=True)
    dec = Decoder(space, embedder, llm, rounds)
    results, todo = [], []
    for v, x in zip(voids, X):
        path = cache / f"{v['id']}.json"
        if path.exists() and not fresh:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            todo.append((v, x))
    print(f"ideas: {len(results)} cached, decoding {len(todo)} voids x {rounds} rounds with {llm.label}")

    def work(v, x):
        r = dec.decode(v, x)
        r["vec"] = r.pop("_vec").round(5).tolist()
        (cache / f"{v['id']}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, v, x): v["id"] for v, x in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                results.append(r)
                print(f"  [{n}/{len(todo)}] {futs[f]}: {r['idea']['title']!r}  novelty {r['novelty']:.3f}"
                      f"  target {r['target_sim']:.3f}  {'landed' if r['landed'] else 'near ' + r['neighbors'][0]['name']}")
            except (LLMError, json.JSONDecodeError, KeyError) as e:
                print(f"  [{n}/{len(todo)}] {futs[f]}: failed ({e})")
    return results


def rank(results: list[dict], dedupe_sim: float = 0.92) -> list[dict]:
    """Rank by novelty against Steam; drop ideas that near-duplicate a higher-ranked idea."""
    ranked = []
    for r in sorted(results, key=lambda r: -r["novelty"]):
        v = np.asarray(r["vec"], dtype=np.float32)
        sims = [float(v @ np.asarray(o["vec"], dtype=np.float32)) for o in ranked]
        if sims and max(sims) > dedupe_sim:
            continue
        r["closest_other_idea"] = max(sims) if sims else None
        ranked.append(r)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked


def calibrate(space: Space, embedder: Embedder, llm, n: int = 16, seed: int = 0) -> dict:
    """Measure how close an LLM *rewrite* of an existing game lands to the original.

    This is the noise floor of the measurement: writing style alone moves a page this far,
    so ideas only count as new to the extent they sit further from every game than this.
    """
    from .voids import anchor_pool

    rng = random.Random(seed)
    picks = rng.sample(list(anchor_pool(space, 50)), n)
    system = ("You are a Steam store copywriter. Rewrite the given game's store page in your own words, "
              "describing the same game accurately. Do not copy sentences.")
    dec = Decoder(space, embedder, llm, rounds=1)
    self_sims, ranks = [], []
    # a snapshot ships without store copy (and added games lack the About text): look these few games up
    g = space.games.iloc[[int(i) for i in picks]]
    if "about" in g and g["about"].map(lambda s: isinstance(s, str)).all():
        texts = {int(a): (s, t) for a, s, t in zip(g["appid"], g["short_description"], g["about"])}
    else:
        texts = store_texts(g["appid"].tolist())

    def work(i):
        g = space.games.iloc[int(i)]
        short, about = texts.get(int(g["appid"]), ("", ""))
        prompt = (f"Game: {g['name']}\nCapsule: {short}\nAbout (excerpt): {about[:1500]}\n"
                  f"Tags: {', '.join(list(g['tags'])[:15])}\n\nWrite the store page fields as JSON. "
                  "Set viability to 5 and leave why_new as a one-line summary.")
        idea = llm.complete_json(system, prompt, IDEA_SCHEMA)
        rec = dec.measure(idea, None)
        own = float(rec["vec"] @ space.E[int(i)].float().cpu().numpy())
        pos = next((k for k, nb in enumerate(rec["neighbors"]) if nb["appid"] == int(g["appid"])), None)
        return own, pos

    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(work, i) for i in picks]):
            try:
                own, pos = f.result()
                self_sims.append(own)
                ranks.append(pos)
            except (LLMError, json.JSONDecodeError, KeyError) as e:
                print(f"  calibration sample failed ({e})")
    out = {
        "n": len(self_sims),
        "median_self_sim": float(np.median(self_sims)),
        "p10_self_sim": float(np.quantile(self_sims, 0.1)),
        "source_is_nearest": sum(1 for r in ranks if r == 0) / max(len(ranks), 1),
        "source_in_top10": sum(1 for r in ranks if r is not None) / max(len(ranks), 1),
    }
    (config.SPACE / "calibration.json").write_text(json.dumps(out, indent=1))
    print(f"calibration: rewrite of a real game lands at median {out['median_self_sim']:.3f} from the original "
          f"(p10 {out['p10_self_sim']:.3f}); original is the nearest game {out['source_is_nearest']:.0%} of the time")
    return out
