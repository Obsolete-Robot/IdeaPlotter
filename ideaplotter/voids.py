"""Find empty regions ("voids") of the game space.

A void target is a point x = normalize(sum_j w_j * a_j): a convex blend of a few anchor
games a_j, so it stays inside the region real game descriptions occupy (interpolation,
not extrapolation into meaningless directions of the 1024-d sphere). The blend weights
w = softmax(theta) are optimised by gradient descent to minimise a soft maximum of the
cosine similarity between x and *every* game on Steam:

    loss(x) = tau * log sum_i exp(cos(x, e_i) / tau)      (-> max_i cos(x, e_i) as tau -> 0)

Because the anchors are themselves in the corpus, the optimum is pushed away from each
anchor and from everything near it, settling where the blend is furthest from any game.

Coherence. Averaging unrelated games cancels what is specific about each of them: the
blend of a subway simulator, a JRPG and a card game is a generic "indie game" vector
that sits far from every real game only because real games are specific. The length of
the un-normalised blend, ||sum_j w_j a_j|| (1 for a single game, smaller the more the
ingredients disagree), measures that loss. Anchors must therefore be related but
different (pairwise cosine in [min_pair_sim, max_pair_sim]) and the loss adds
    lam * relu(min_coherence - ||sum_j w_j a_j||)
so voids stay as specific as real games. Candidates are ranked by
novelty = 1 - max_i cos(x, e_i) and greedily filtered so the kept targets are far apart too.
"""
import json

import numpy as np
import torch

from . import config
from .space import Space


def anchor_pool(space: Space, min_reviews: int) -> np.ndarray:
    g = space.games
    ok = (~g["nsfw"]) & (g["review_count"].fillna(0) >= min_reviews)
    return np.flatnonzero(ok.to_numpy())


def sample_anchor_sets(space: Space, n: int, m_lo: int, m_hi: int, pool: np.ndarray, min_pair_sim: float,
                       max_pair_sim: float, rng: np.random.Generator, distinct_clusters: bool = True):
    """n anchor sets of m_lo..m_hi games, pairwise related but not near-duplicates (globally also from distinct clusters).

    Anchors after the first are drawn from games whose similarity to the first is in range,
    so sets of related games are found quickly even in a pool of 50k+ games."""
    idx = np.zeros((n, m_hi), dtype=np.int64)
    mask = np.zeros((n, m_hi), dtype=bool)
    P = space.E[torch.from_numpy(pool).to(space.device)].float().cpu().numpy()
    clusters = space.cluster[pool]
    for r in range(n):
        m = rng.integers(m_lo, m_hi + 1)
        first = int(rng.integers(len(pool)))
        s0 = P @ P[first]
        cand = np.flatnonzero((s0 >= min_pair_sim) & (s0 <= max_pair_sim))
        chosen = [first]  # positions in pool
        for _ in range(50 * m):
            if len(chosen) == m or not len(cand):
                break
            j = int(cand[rng.integers(len(cand))])
            if j in chosen or (distinct_clusters and any(clusters[j] == clusters[c] for c in chosen)):
                continue
            sims = P[chosen] @ P[j]
            if sims.max() > max_pair_sim or sims.min() < min_pair_sim:
                continue
            chosen.append(j)
        idx[r, : len(chosen)] = pool[chosen]
        mask[r, : len(chosen)] = True
    keep = mask.sum(1) >= 2
    return idx[keep], mask[keep]


def optimize(space: Space, idx: np.ndarray, mask: np.ndarray, steps: int = 200, tau: float = 0.015,
             lr: float = 0.08, batch: int = 512, seed: int = 0, min_coherence: float = 0.82, lam: float = 2.0,
             min_share: float = 0.0, verbose: bool = True):
    """Gradient-descend blend weights so each blend moves as far as possible from all games while staying coherent.

    min_share reserves that fraction of the total weight, split evenly, so every anchor keeps
    at least min_share / m of the blend (used when the anchors were chosen by a person)."""
    dev = space.device
    E = space.E
    g = torch.Generator(device="cpu").manual_seed(seed)
    out_x, out_w, out_c = [], [], []
    for s in range(0, len(idx), batch):
        I = torch.from_numpy(idx[s : s + batch]).to(dev)
        Mk = torch.from_numpy(mask[s : s + batch]).to(dev)
        A = E[I].float()                                            # B x M x D
        theta = (0.3 * torch.randn(I.shape, generator=g)).to(dev).requires_grad_(True)
        floor = Mk.float() * (min_share / Mk.sum(1, keepdim=True).clamp(min=1))
        weights = lambda: floor + (1 - min_share) * torch.softmax(theta.masked_fill(~Mk, -1e9), dim=1)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            w = weights()
            b = (w.unsqueeze(-1) * A).sum(1)
            coherence = b.norm(dim=1)
            x = b / coherence.unsqueeze(1)
            S = (x.to(E.dtype) @ E.T).float()                       # B x N cosine to every game
            loss = (tau * torch.logsumexp(S / tau, dim=1) + lam * torch.relu(min_coherence - coherence)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            w = weights()
            b = (w.unsqueeze(-1) * A).sum(1)
            out_c.append(b.norm(dim=1).cpu().numpy())
            x = torch.nn.functional.normalize(b, dim=1)
        out_x.append(x.cpu().numpy())
        out_w.append(w.cpu().numpy())
        if verbose:
            print(f"  optimised {min(s + batch, len(idx))}/{len(idx)} blends")
    return np.concatenate(out_x), np.concatenate(out_w), np.concatenate(out_c)


def select_diverse(X: np.ndarray, novelty: np.ndarray, idx: np.ndarray, w: np.ndarray, top: int,
                   max_sim: float, max_anchor_uses: int) -> list[int]:
    """Greedy: best novelty first, skipping targets too close to an already chosen one or reusing anchors."""
    chosen, uses = [], {}
    for r in np.argsort(-novelty):
        if novelty[r] < 0:  # rejected (incoherent) candidates
            break
        if chosen and float((X[chosen] @ X[r]).max()) > max_sim:
            continue
        main = [int(a) for a, wt in zip(idx[r], w[r]) if wt >= 0.1]
        if any(uses.get(a, 0) >= max_anchor_uses for a in main):
            continue
        chosen.append(int(r))
        for a in main:
            uses[a] = uses.get(a, 0) + 1
        if len(chosen) == top:
            break
    return chosen


def describe_target(space: Space, x: np.ndarray, anchors: list[tuple[int, float]], coherence: float,
                    k: int = 8) -> dict:
    sims, nbr = space.neighbors(x, k)
    sims, nbr = sims[0], nbr[0]
    p = space.tag_probs(x)[0]
    p_nbr = space.tag_probs(space.E[torch.from_numpy(nbr).to(space.device)].float().cpu().numpy()).mean(0)
    # rank tags by how characteristic they are of this point, not by raw frequency ("Indie" is everywhere)
    lift = p * np.log((p + 0.01) / (space.tag_base + 0.01))
    top_tags = [t for t in np.argsort(-lift) if p[t] >= 0.15][:15]
    delta = p - p_nbr
    return {
        "nearest_sim": float(sims[0]),
        "novelty": float(1 - sims[0]),
        "isolation_pct": float(space.isolation_percentile(float(sims[0]))),
        "coherence": round(float(coherence), 3),
        "anchors": [dict(space.game_card(a), weight=round(float(wt), 3),
                         anchor_sim=round(float(x @ space.E[a].float().cpu().numpy()), 3),
                         cluster=space.cluster_names[space.cluster[a]]) for a, wt in anchors],
        "neighbors": [dict(space.game_card(j), sim=round(float(s), 3)) for s, j in zip(sims, nbr)],
        "predicted_tags": [(space.tag_vocab[t], round(float(p[t]), 3)) for t in top_tags],
        "more_than_neighbors": [space.tag_vocab[t] for t in np.argsort(-delta)[:6] if delta[t] > 0.05],
        "less_than_neighbors": [space.tag_vocab[t] for t in np.argsort(delta)[:6] if delta[t] < -0.05],
    }


def _anchors_of(idx, W, mask, r):
    """Anchor games and blend weights of candidate r (games the optimiser dropped to ~0% are left out)."""
    return sorted([(int(a), float(wt)) for a, wt, m in zip(idx[r], W[r], mask[r]) if m and wt >= 0.03],
                  key=lambda t: -t[1])


def find(space: Space, n_candidates: int = 4096, anchors: tuple[int, int] = (2, 4), top: int = 40,
         min_reviews: int = 20, min_pair_sim: float = 0.3, max_pair_sim: float = 0.8, min_coherence: float = 0.82,
         diversity: float = 0.88, max_anchor_uses: int = 1, steps: int = 200, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = anchor_pool(space, min_reviews)
    print(f"voids: sampling {n_candidates} anchor sets of {anchors[0]}-{anchors[1]} related games "
          f"from {len(pool):,} eligible anchors (>= {min_reviews} reviews, SFW)")
    idx, mask = sample_anchor_sets(space, n_candidates, anchors[0], anchors[1], pool, min_pair_sim, max_pair_sim, rng)
    X, W, C = optimize(space, idx, mask, steps=steps, seed=seed, min_coherence=min_coherence)

    sims, _ = space.neighbors(X, 1)
    novelty = 1 - sims[:, 0]
    novelty[C < min_coherence - 0.02] = -1
    ok = novelty > -1
    q = np.quantile(novelty[ok], [0.5, 0.9, 0.99])
    print(f"  {ok.sum()} coherent blends; novelty (1 - nearest-game cosine): median {q[0]:.3f}, "
          f"p90 {q[1]:.3f}, p99 {q[2]:.3f}; typical Steam game: {1 - np.median(space.isolation_ref):.3f}")

    chosen = select_diverse(X, novelty, idx, W, top, diversity, max_anchor_uses)
    voids = []
    for rank, r in enumerate(chosen, 1):
        v = describe_target(space, X[r], _anchors_of(idx, W, mask, r), C[r])
        v["id"] = f"v{rank:03d}"
        voids.append(v)

    config.SPACE.mkdir(parents=True, exist_ok=True)
    np.save(config.SPACE / "void_x.npy", X[chosen].astype(np.float32))
    (config.SPACE / "voids.json").write_text(json.dumps(voids, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  kept {len(voids)} diverse voids -> {config.SPACE / 'voids.json'}")
    return voids


def pick_best(space: Space, X: np.ndarray, C: np.ndarray, min_coherence: float, exclude=None) -> int:
    """Most novel candidate, preferring coherent ones and skipping voids already explored (cosine > 0.9)."""
    sims, _ = space.neighbors(X, 1)
    score = 1 - sims[:, 0]
    score[C < min_coherence - 0.02] -= 1
    if exclude is not None and len(exclude):
        score[(X @ np.asarray(exclude, dtype=np.float32).T).max(1) > 0.9] -= 2
    return int(np.argmax(score))


def _result(space, idx, W, mask, X, C, r, **extra):
    v = describe_target(space, X[r], _anchors_of(idx, W, mask, r), C[r])
    v.update(extra)
    return v, X[r]


def find_in_region(space: Space, rows: np.ndarray, anchors: tuple[int, int] = (2, 4), n_candidates: int = 768,
                   exclude: np.ndarray | None = None, min_reviews: int = 5, min_coherence: float = 0.82,
                   steps: int = 150, seed: int | None = None) -> tuple[dict, np.ndarray]:
    """The emptiest coherent blend of games inside a map selection.

    Anchors come only from the selected games, so the target lies inside that region's part
    of the space; novelty is still measured against every game on Steam. Targets within
    0.9 cosine of `exclude` (voids already explored) are skipped so repeat requests differ.
    """
    rows = np.asarray(rows, dtype=np.int64)
    sfw = rows[~space.games["nsfw"].to_numpy()[rows]]
    reviewed = sfw[space.games["review_count"].fillna(0).to_numpy()[sfw] >= min_reviews]
    pool = reviewed if len(reviewed) >= 12 else sfw
    if len(pool) < 2:
        raise ValueError("Select an area that contains at least two games.")
    rng = np.random.default_rng(seed)
    hi = min(anchors[1], len(pool))
    idx, mask = sample_anchor_sets(space, n_candidates, min(anchors[0], hi), hi, pool, 0.25, 0.9, rng,
                                   distinct_clusters=False)
    if not len(idx):
        raise ValueError("The games in this area are too unrelated to blend; try a tighter selection.")
    X, W, C = optimize(space, idx, mask, steps=steps, seed=int(rng.integers(1 << 30)), batch=768,
                       min_coherence=min_coherence, verbose=False)
    r = pick_best(space, X, C, min_coherence, exclude)
    return _result(space, idx, W, mask, X, C, r, pool_size=int(len(pool)), candidates=int(len(X)))


def find_for_blend(space: Space, rows: list[int], exclude=None, min_coherence: float = 0.72,
                   restarts: int = 192, seed: int | None = None) -> tuple[dict, np.ndarray]:
    """The mix of a person's chosen games that sits furthest from every game on Steam, the chosen ones included.

    Every chosen game keeps at least half of an equal share of the blend, so the result is a
    combination of all of them; the optimiser decides the rest of the weighting."""
    rows = list(dict.fromkeys(int(r) for r in rows))
    if not 2 <= len(rows) <= 8:
        raise ValueError("Pick between 2 and 8 games to combine.")
    rng = np.random.default_rng(seed)
    idx = np.tile(np.array(rows, dtype=np.int64), (restarts, 1))
    mask = np.ones_like(idx, dtype=bool)
    X, W, C = optimize(space, idx, mask, steps=200, seed=int(rng.integers(1 << 30)), batch=restarts,
                       min_coherence=min_coherence, min_share=0.5, verbose=False)
    r = pick_best(space, X, C, min_coherence, exclude)
    names = [space.games["name"].iloc[i] for i in rows]
    return _result(space, idx, W, mask, X, C, r, pool_size=len(rows), candidates=restarts,
                   context="Someone chose these games to combine: " + ", ".join(names) + ". The point is the mix "
                           "of them that sits furthest from every game on Steam, including these. The game "
                           "should clearly draw on all of them while being none of them.")


def find_global_one(space: Space, exclude=None, n_candidates: int = 1024, anchors: tuple[int, int] = (2, 5),
                    min_coherence: float = 0.75, seed: int | None = None) -> tuple[dict, np.ndarray]:
    """The emptiest blend anywhere on Steam (looser coherence than the batch search, for wilder results)."""
    rng = np.random.default_rng(seed)
    pool = anchor_pool(space, 20)
    idx, mask = sample_anchor_sets(space, n_candidates, anchors[0], anchors[1], pool, 0.2, 0.8, rng)
    X, W, C = optimize(space, idx, mask, steps=150, seed=int(rng.integers(1 << 30)), batch=512,
                       min_coherence=min_coherence, verbose=False)
    r = pick_best(space, X, C, min_coherence, exclude)
    return _result(space, idx, W, mask, X, C, r, pool_size=int(len(pool)), candidates=int(len(X)))


def load() -> tuple[list[dict], np.ndarray]:
    voids = json.loads((config.SPACE / "voids.json").read_text(encoding="utf-8"))
    return voids, np.load(config.SPACE / "void_x.npy")
