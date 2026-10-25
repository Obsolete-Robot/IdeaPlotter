"""Geometry of the game space: nearest neighbours, a tag probe, clusters and a 2-D map.

`analyze()` computes everything once and saves it under data/space/; `Space` loads it
back (embeddings live on the GPU) for the void search and the idea decoder.
"""
import json

import numpy as np
import pandas as pd
import torch

from . import config

K_NEIGHBORS = 30


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_dtype(device: str) -> torch.dtype:
    """float16 on GPUs that compute it fast (Volta / RTX 20 and newer); Pascal cards such as the
    GTX 10 series run it at a small fraction of their float32 speed, and CPUs want float32 too."""
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
        return torch.float16
    return torch.float32


def _as_tensor(E: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(E)).to(device=device, dtype=compute_dtype(device))


@torch.no_grad()
def topk(Q: torch.Tensor, E: torch.Tensor, k: int, exclude_offset: int | None = None, chunk: int = 2048):
    """Cosine top-k of each query row against E. If exclude_offset is given, Q is E[offset:...] and self-matches are skipped."""
    sims, idxs = [], []
    for i in range(0, len(Q), chunk):
        S = Q[i : i + chunk].to(E.dtype) @ E.T
        if exclude_offset is not None:
            rows = torch.arange(S.shape[0], device=S.device)
            S[rows, rows + exclude_offset + i] = -2
        s, j = S.topk(k, dim=1)
        sims.append(s.float().cpu())
        idxs.append(j.cpu())
    return torch.cat(sims).numpy(), torch.cat(idxs).numpy()


def train_probe(E: torch.Tensor, tag_lists, min_count: int = 40, epochs: int = 12, seed: int = 0):
    """Multi-label linear probe: embedding -> P(tag). Lets us read predicted metadata off any point in space."""
    counts = {}
    for ts in tag_lists:
        for t in ts:
            counts[t] = counts.get(t, 0) + 1
    vocab = sorted(t for t, c in counts.items() if c >= min_count)
    col = {t: i for i, t in enumerate(vocab)}
    Y = torch.zeros(len(tag_lists), len(vocab), dtype=torch.float16)
    for r, ts in enumerate(tag_lists):
        for t in ts:
            if t in col:
                Y[r, col[t]] = 1

    dev = E.device
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(Y), generator=g)
    n_val = max(2000, len(Y) // 20)
    val, tr = perm[:n_val], perm[n_val:]
    Y = Y.to(dev)
    lin = torch.nn.Linear(E.shape[1], len(vocab)).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=1e-4)
    scale = 20.0  # unit vectors need a temperature for the logits to reach useful ranges
    for ep in range(epochs):
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), 4096):
            b = order[i : i + 4096].to(dev)
            logits = lin(E[b].float() * scale)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y[b].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        P = torch.sigmoid(lin(E[val.to(dev)].float() * scale)).cpu().numpy()
    Yv = Y[val.to(dev)].float().cpu().numpy()
    from sklearn.metrics import average_precision_score

    aps = [average_precision_score(Yv[:, j], P[:, j]) for j in range(len(vocab)) if Yv[:, j].sum() > 0]
    print(f"tag probe: {len(vocab)} tags, held-out mean AP {np.mean(aps):.3f} (tag base rate {Yv.mean():.3f})")
    W = (lin.weight.detach().float().cpu().numpy() * scale).T  # D x T
    b = lin.bias.detach().float().cpu().numpy()
    return vocab, W.astype(np.float32), b.astype(np.float32)


@torch.no_grad()
def spherical_kmeans(E: torch.Tensor, k: int = 200, iters: int = 30, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    C = E[torch.randperm(len(E), generator=g)[:k].to(E.device)].float()
    for _ in range(iters):
        assign = torch.cat([(E[i : i + 8192].float() @ C.T).argmax(1) for i in range(0, len(E), 8192)])
        newC = torch.zeros_like(C).index_add_(0, assign, E.float())
        sizes = torch.bincount(assign, minlength=k)
        empty = sizes == 0
        if empty.any():
            newC[empty] = E[torch.randint(len(E), (int(empty.sum()),), generator=g).to(E.device)].float()
        C = torch.nn.functional.normalize(newC, dim=1)
    return assign.cpu().numpy(), C.cpu().numpy()


def name_clusters(assign: np.ndarray, tag_lists, k: int) -> list[str]:
    """Label each cluster by the tags most over-represented in it (support * log-lift)."""
    n = len(tag_lists)
    glob, per = {}, [dict() for _ in range(k)]
    for c, ts in zip(assign, tag_lists):
        for t in ts[:10]:
            glob[t] = glob.get(t, 0) + 1
            per[c][t] = per[c].get(t, 0) + 1
    sizes = np.bincount(assign, minlength=k)
    names = []
    for c in range(k):
        scored = []
        for t, cnt in per[c].items():
            p_c, p = cnt / max(sizes[c], 1), glob[t] / n
            if p_c >= 0.15:
                scored.append((p_c * np.log(p_c / p), t))
        names.append(" / ".join(t for _, t in sorted(scored, reverse=True)[:3]) or "misc")
    return names


def umap_layout(E_np: np.ndarray, nn_sim: np.ndarray, nn_idx: np.ndarray, seed: int = 0, dims: int = 2) -> np.ndarray:
    import umap

    n = len(E_np)
    k = 15
    knn_idx = np.concatenate([np.arange(n)[:, None], nn_idx[:, : k - 1]], axis=1).astype(np.int64)
    knn_dist = np.concatenate([np.zeros((n, 1)), np.clip(1 - nn_sim[:, : k - 1], 0, 2)], axis=1).astype(np.float32)
    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=k,
        min_dist=0.08,
        metric="cosine",
        precomputed_knn=(knn_idx, knn_dist, None),
        random_state=seed,
        low_memory=True,
        verbose=False,
    )
    return reducer.fit_transform(E_np).astype(np.float32)


def load_aligned(full: bool = False):
    """The games table reordered to match the rows of emb.npy, plus the embedding model name.

    Uses the full table from `build` when there is one, else the slim table from a snapshot.
    full=True insists on the full table, which has the long store text."""
    meta = json.loads((config.SPACE / "emb_meta.json").read_text())
    sources = [config.GAMES] if full else [p for p in (config.GAMES, config.SNAPSHOT_GAMES) if p.exists()]
    for path in sources:
        if not path.exists():
            break
        games = pd.read_parquet(path)
        order = pd.Index(games["appid"]).get_indexer(meta["appids"])
        if (order >= 0).all():
            return games.iloc[order].reset_index(drop=True), meta["model"]
    raise RuntimeError("no games table matches emb.npy: run `python -m ideaplotter download` (prebuilt snapshot) "
                       "or `fetch` + `build`")


def analyze(k_clusters: int = 200, make_map: bool = True) -> None:
    df, _ = load_aligned()
    dev = _device()
    E_np = np.load(config.SPACE / "emb.npy").astype(np.float32)
    E = _as_tensor(E_np, dev)
    tag_lists = df["tags"].map(list).tolist()

    print("neighbours: exact top-30 for every game")
    nn_sim, nn_idx = topk(E, E, K_NEIGHBORS, exclude_offset=0)
    np.save(config.SPACE / "nn_sim.npy", nn_sim.astype(np.float16))
    np.save(config.SPACE / "nn_idx.npy", nn_idx.astype(np.int32))
    q = np.quantile(nn_sim[:, 0], [0.05, 0.25, 0.5, 0.75, 0.95])
    print("  nearest-neighbour similarity quantiles (5/25/50/75/95%): " + "  ".join(f"{v:.3f}" for v in q))

    vocab, W, b = train_probe(E, tag_lists)
    np.savez(config.SPACE / "probe.npz", W=W, b=b, vocab=np.array(vocab, dtype=object))

    print(f"clusters: spherical k-means, k={k_clusters}")
    assign, C = spherical_kmeans(E, k_clusters)
    names = name_clusters(assign, tag_lists, k_clusters)
    np.save(config.SPACE / "cluster.npy", assign.astype(np.int16))
    np.save(config.SPACE / "centroids.npy", C.astype(np.float32))
    (config.SPACE / "cluster_names.json").write_text(json.dumps(names, indent=0), encoding="utf-8")

    if make_map:
        print("map: UMAP 2-D layout from the exact neighbour graph")
        xy = umap_layout(E_np, nn_sim, nn_idx)
        np.save(config.SPACE / "xy.npy", xy)
        print("map: UMAP 3-D layout for the explorer")
        np.save(config.SPACE / "xyz.npy", umap_layout(E_np, nn_sim, nn_idx, dims=3))


class Space:
    """Loaded game space: corpus table, GPU embeddings, neighbour stats, probe, clusters, map."""

    def __init__(self):
        self.games, self.model_name = load_aligned()
        self.device = _device()
        self.E = _as_tensor(np.load(config.SPACE / "emb.npy"), self.device)
        self.nn_sim = np.load(config.SPACE / "nn_sim.npy").astype(np.float32)
        self.nn_idx = np.load(config.SPACE / "nn_idx.npy")
        probe = np.load(config.SPACE / "probe.npz", allow_pickle=True)
        self.probe_W, self.probe_b, self.tag_vocab = probe["W"], probe["b"], list(probe["vocab"])
        self.cluster = np.load(config.SPACE / "cluster.npy").astype(np.int64)
        self.cluster_names = json.loads((config.SPACE / "cluster_names.json").read_text(encoding="utf-8"))
        xy_path = config.SPACE / "xy.npy"
        self.xy = np.load(xy_path) if xy_path.exists() else None
        if self.xy is not None:  # map coordinates quantised to 0..65535, shared with the browser
            self.q_lo, self.q_hi = self.xy.min(0), self.xy.max(0)
            self.xy_q = ((self.xy - self.q_lo) / (self.q_hi - self.q_lo) * 65535).astype(np.float32)
        # the explorer draws the 3-D layout; the 2-D one only resolves areas and holes saved by older versions
        xyz_path = config.SPACE / "xyz.npy"
        self.xyz = np.load(xyz_path) if xyz_path.exists() else None
        if self.xyz is not None:  # one scale for all three axes, so distances on the map stay isotropic
            lo, hi = self.xyz.min(0), self.xyz.max(0)
            self.p_scale = float((hi - lo).max()) / 65535
            self.p_lo = (lo + hi) / 2 - self.p_scale * 65535 / 2
            self.xyz_q = ((self.xyz - self.p_lo) / self.p_scale).astype(np.float32)
        self.isolation_ref = np.sort(self.nn_sim[:, 0])  # every game's similarity to its closest other game
        self.all_tags = sorted({t for ts in self.games["tags"] for t in ts})
        col = {t: i for i, t in enumerate(self.tag_vocab)}
        counts = np.zeros(len(self.tag_vocab))
        for ts in self.games["tags"]:
            for t in ts:
                if t in col:
                    counts[col[t]] += 1
        self.tag_base = counts / len(self.games)  # share of games carrying each probe tag
        self.far_ok = self._far_ok(self.games)
        self.n_snapshot = len(self.games)
        self._load_added()

    @staticmethod
    def _far_ok(games: pd.DataFrame) -> np.ndarray:
        """"Furthest games" only lists known games with real store text; otherwise a handful of empty
        or broken pages would be the furthest thing from everything."""
        if "text_len" not in games:
            games = games.assign(text_len=games["short_description"].str.len() + games["about"].str.len())
        return (games["review_count"].fillna(0).to_numpy() >= 50) & (games["text_len"].to_numpy() >= 300)

    def _load_added(self) -> None:
        path = config.ADDED / "games.parquet"
        if not path.exists():
            return
        df, V = pd.read_parquet(path), np.load(config.ADDED / "emb.npy")
        meta = json.loads((config.ADDED / "meta.json").read_text())
        if meta.get("model") != self.model_name:
            print(f"  ignoring {len(df)} added games: embedded with {meta.get('model')}, not {self.model_name}")
            return
        new = ~df["appid"].isin(self.games["appid"])  # a newer snapshot may already have them
        if new.any():
            self._append(df[new.to_numpy()].reset_index(drop=True), V[new.to_numpy()])

    def add_game(self, row: dict, vec: np.ndarray) -> int:
        """Add one game looked up after the snapshot was built. It is placed among its nearest games
        and joins every measurement from now on. Saved under data/space/added/, so it stays."""
        df = pd.DataFrame([row])
        self._append(df, vec[None, :])
        config.ADDED.mkdir(parents=True, exist_ok=True)
        path = config.ADDED / "games.parquet"
        self.games.iloc[self.n_snapshot :][list(df.columns)].to_parquet(path, index=False)
        np.save(config.ADDED / "emb.npy", self.E[self.n_snapshot :].float().cpu().numpy().astype(np.float16))
        (config.ADDED / "meta.json").write_text(json.dumps({"model": self.model_name}))
        return len(self.games) - 1

    @torch.no_grad()
    def _append(self, df: pd.DataFrame, V: np.ndarray) -> None:
        """Append games to every per-game array. Map positions and clusters come from their nearest games;
        the whole-map statistics (isolation scale, probe, cluster names) stay those of the snapshot."""
        V = V.astype(np.float32)
        n0 = len(self.games)
        if self.xy is not None:
            xy = self.place_2d(V)
            self.xy = np.concatenate([self.xy, xy.astype(self.xy.dtype)])
            self.xy_q = np.concatenate([self.xy_q, ((xy - self.q_lo) / (self.q_hi - self.q_lo) * 65535).astype(np.float32)])
        if self.xyz is not None:
            q = np.asarray(self.place_3d(V), dtype=np.float32)
            self.xyz_q = np.concatenate([self.xyz_q, q])
            self.xyz = np.concatenate([self.xyz, (q * self.p_scale + self.p_lo).astype(self.xyz.dtype)])
        C = np.load(config.SPACE / "centroids.npy")
        self.cluster = np.concatenate([self.cluster, (V @ C.T).argmax(1).astype(np.int64)])
        self.E = torch.cat([self.E, _as_tensor(V, self.device)])
        sims, idx = topk(self.E[n0:], self.E, K_NEIGHBORS, exclude_offset=n0)
        self.nn_sim = np.concatenate([self.nn_sim, sims.astype(np.float32)])
        self.nn_idx = np.concatenate([self.nn_idx, idx.astype(self.nn_idx.dtype)])
        self.games = pd.concat([self.games, df], ignore_index=True)
        self.far_ok = np.concatenate([self.far_ok, self._far_ok(df)])

    def __len__(self):
        return len(self.games)

    def neighbors(self, X: np.ndarray, k: int = 10):
        Q = torch.from_numpy(np.atleast_2d(X).astype(np.float32)).to(self.device)
        return topk(Q, self.E, k)

    @torch.no_grad()
    def closest_and_furthest(self, row: int, k: int = 10) -> dict:
        sims = (self.E @ self.E[int(row)]).float().cpu().numpy()
        sims[int(row)] = -9
        close = np.argsort(-sims)[:k]
        sims[int(row)] = 9
        far = [int(j) for j in np.argsort(sims) if self.far_ok[j]][:k]
        item = lambda j: {"row": int(j), "name": self.games["name"].iloc[j], "appid": int(self.games["appid"].iloc[j]),
                          "sim": round(float(sims[j]), 3)}
        return {"closest": [item(j) for j in close], "furthest": [item(j) for j in far]}

    def tag_probs(self, X: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(X) @ self.probe_W + self.probe_b
        return 1 / (1 + np.exp(-z))

    def isolation_percentile(self, nn_sim: float) -> float:
        """Share of Steam games whose closest neighbour is nearer than this point's closest neighbour."""
        return 1.0 - np.searchsorted(self.isolation_ref, nn_sim) / len(self.isolation_ref)

    def place_2d(self, X: np.ndarray) -> np.ndarray:
        """Approximate map position: similarity-weighted mean of the 2-D positions of the 10 nearest games."""
        sims, idx = self.neighbors(X, 10)
        w = np.exp((sims - sims[:, :1]) / 0.02)
        return (self.xy[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)

    def place_3d(self, X: np.ndarray) -> list:
        """Quantised 3-D map position of each vector, from its 10 nearest games (see place_2d)."""
        sims, idx = self.neighbors(X, 10)
        w = np.exp((sims - sims[:, :1]) / 0.02)
        return ((self.xyz_q[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)).round(1).tolist()

    def to_q(self, xy: np.ndarray) -> list:
        return ((np.asarray(xy) - self.q_lo) / (self.q_hi - self.q_lo) * 65535).round(1).tolist()

    def rows_in_polygon(self, poly) -> np.ndarray:
        """Rows whose quantised map position lies inside the polygon (even-odd ray casting)."""
        P = np.asarray(poly, dtype=np.float32)
        lo, hi = P.min(0), P.max(0)
        cand = np.flatnonzero(((self.xy_q >= lo) & (self.xy_q <= hi)).all(1))
        x, y = self.xy_q[cand, 0], self.xy_q[cand, 1]
        inside = np.zeros(len(cand), dtype=bool)
        for (x1, y1), (x2, y2) in zip(P, np.roll(P, -1, axis=0)):
            crosses = (y1 > y) != (y2 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (x < xi)
        return cand[inside]

    def summarize(self, rows: np.ndarray) -> dict:
        g = self.games.iloc[rows]
        tag_counts = {}
        for ts in g["tags"]:
            for t in list(ts)[:10]:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        clusters = np.bincount(self.cluster[rows], minlength=len(self.cluster_names))
        notable = g.sort_values("review_count", ascending=False).head(8)
        return {
            "count": int(len(rows)),
            "tags": [(t, round(c / len(rows), 3)) for t, c in sorted(tag_counts.items(), key=lambda kv: -kv[1])[:16]],
            "clusters": [(self.cluster_names[c], round(int(clusters[c]) / len(rows), 3))
                         for c in np.argsort(-clusters)[:4] if clusters[c]],
            "notable": [{"name": r["name"], "appid": int(r["appid"]), "reviews": int(r["review_count"] or 0)}
                        for _, r in notable.iterrows()],
            "nn_median": float(np.median(self.nn_sim[rows, 0])),
        }

    def game_card(self, i: int, about_chars: int = 0) -> dict:
        g = self.games.iloc[int(i)]
        card = {
            "appid": int(g["appid"]),
            "name": g["name"],
            # snapshots ship without store copy: "" until looked up (decode.fill_short, /api/store)
            "short_description": g["short_description"] if isinstance(g.get("short_description"), str) else "",
            "tags": list(g["tags"])[:12],
            "reviews": int(g["review_count"] or 0),
            "url": f"https://store.steampowered.com/app/{int(g['appid'])}",
        }
        if about_chars and isinstance(g.get("about"), str):  # snapshots ship without the long store text
            card["about"] = g["about"][:about_chars]
        return card
