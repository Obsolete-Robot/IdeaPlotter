# IdeaPlotter

Plot every game on Steam in a semantic vector space, find the empty places, and turn them into new game ideas, ranked by how unlike every existing game they are.

**Try it:** [obsoleterobot.com/ideaplotter](https://obsoleterobot.com/ideaplotter/) · **Watch:** [what you're looking at (YouTube)](https://youtu.be/ObvISKp1dWo)

Every point of light on the map is a Steam game, placed by what its store page says, so similar games sit together. The dark gaps between them are games nobody has made yet. Pick a gap, a region or a few games you like, and IdeaPlotter finds the emptiest spot there, has an AI write the game that would live in it, then measures where that game actually lands and revises it.

## Install

You need:

- **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/); on Windows, tick "Add python.exe to PATH")
- **About 4 GB of disk space**
- **Optional: an NVIDIA graphics card.** It finds empty spots in seconds; on the CPU each search takes a few minutes.
- **Optional: an AI to write ideas.** An [OpenRouter](https://openrouter.ai/keys) or [Anthropic](https://console.anthropic.com) API key, or [Claude Code](https://claude.com/claude-code). You can browse the map right away and add one later.

Download the project (**Code → Download ZIP** on GitHub and unzip it, or `git clone https://github.com/Obsolete-Robot/IdeaPlotter.git`), then run the installer:

- **Windows:** double-click **`install.bat`**
- **macOS / Linux:** run `./install.sh` in a terminal

The installer:

1. creates a Python environment in `.venv/`;
2. installs PyTorch to match your computer (CUDA for an NVIDIA card, otherwise the CPU build) and the other packages;
3. downloads the prebuilt map of every Steam game (about 440 MB);
4. asks how ideas should be written, and saves your key (if you give one) to `.env` in the project folder.

It is safe to run again at any time: finished steps are skipped, and it is how you change or add your key later.

## Run

- **Windows:** double-click **`run.bat`**
- **macOS / Linux:** `./run.sh`

The map opens in your browser at http://127.0.0.1:8765. Keep the window open while you use it; close it or press Ctrl+C to stop. The first start also downloads the embedding model (about 1.2 GB); after that, starting takes about 30 seconds.

## Using the map

The map is a 3-D cloud: drag to orbit, right-drag (or Alt-drag) to pan, scroll to zoom. **Drift** slowly turns it while nothing is selected. **2D / 3D** at the top (or the 2 and 3 keys) switches to a flat map and back. There are four ways to find empty space:

- **Holes.** The dashed rings are empty pockets with games on every side. Click one, or any empty spot, then **Write a game for this gap**. These produce the most novel ideas.
- **Areas.** Press **L** and draw around a region. Shift-drag adds games to the selection and Ctrl-drag removes them. Then **Find a void here and write a game**.
- **Games.** Search for a game or click its point to see its closest and furthest games. **Add to blend** collects 2–8 games, and **Find the most novel mix and write a game** finds the mix of them furthest from every game on Steam.
- **Anywhere.** **Find the emptiest spot anywhere** searches all of Steam.

Each ends the same way: the AI writes a game for that point, and you watch each draft get measured and revised. Ideas are saved in `out/ideas/` and appear in the list on the right.

The map holds Steam's catalog as of October 2026. For newer games, type a name in the search box, choose **Search Steam**, then **add to map** to place one.

## Choosing the AI writer

| Writer | Set up with | Cost and speed per idea |
|---|---|---|
| OpenRouter, Claude Sonnet 5.5 (default) | an OpenRouter key | about $0.03, 20 seconds; wrote the most original ideas in a side-by-side test |
| OpenRouter, any other model | an OpenRouter key, plus `--model` | e.g. `anthropic/claude-haiku-5.5`: about $0.002 |
| Anthropic API, Claude Opus 5.5 | an Anthropic key | more per idea; highest effort |
| Claude Code | your [Claude Code](https://claude.com/claude-code) login | uses your Claude plan |

To switch, run the installer again, or edit `.env`. To try another OpenRouter model for one session: `run.bat --model google/gemini-3.8-flash` (or `./run.sh --model ...`). Any [OpenRouter model](https://openrouter.ai/models) that supports structured outputs works.

## Command line

Everything the map does is also available as commands. Use `.venv\Scripts\python` on Windows and `.venv/bin/python` on macOS/Linux:

```bash
.venv/bin/python -m ideaplotter serve --open        # the map (what run.bat / run.sh start)
.venv/bin/python -m ideaplotter voids --top 40      # batch: the 40 emptiest coherent spots on Steam
.venv/bin/python -m ideaplotter ideas --limit 12    # write games for them -> out/ideas.md, out/explorer.html
.venv/bin/python -m ideaplotter calibrate           # optional: measure the rewrite noise floor
.venv/bin/python -m ideaplotter --help              # everything else
```

Outputs:

- `out/ideas.md`: ranked ideas with metrics, recipe, closest games and draft history
- `out/ideas.json`: the same, machine-readable
- `out/explorer.html`: a self-contained copy of the map with all ideas (browsing works; writing needs the running app)
- `out/ideas/*.json`: one file per idea, the cache everything else is built from

## Technical details

```
Steam catalog ──► store pages ──► embeddings ──► find voids ──► LLM writes a game ──► embed it, measure, revise
  ~195k games      clean text      1024-d          (math)        for each void          rank by novelty
```

### How it works

**1. Data.** Steam's public store API (`IStoreQueryService/Query`, no key, 1000 games per call) provides each game's short description, store text, weighted user tags, reviews and release data. Stubs, games with fewer than 3 tags and non-Latin-script pages are dropped, which leaves about 189k games. Unreleased "coming soon" pages stay in, because they are ideas someone already had.

**2. Embedding.** Each game becomes one document (`title / short description / Tags: ... / first 800 chars of About`), embedded with `Qwen/Qwen3-Embedding-0.6B` (1024-d, cosine). Generated ideas are embedded in exactly the same format, so they are measured on the same scale as real games.

**3. Geometry:**
- exact top-30 neighbours of every game, giving the distribution of "how close is each game's nearest twin";
- a linear **tag probe** (embedding → P(tag) for ~400 tags), so metadata can be read off any point in space, including the empty ones;
- 200 spherical k-means clusters, named by their most over-represented tags;
- UMAP layouts (2-D and 3-D) built from the exact neighbour graph. They are only for display; every measurement uses the full 1024-d space.

**4. Voids.** A void target is a convex blend of a few real games, `x = normalize(Σ wⱼ aⱼ)`. Keeping it inside the region real store pages occupy avoids extrapolating into meaningless directions. The weights are optimised by gradient descent to minimise a soft maximum of the cosine similarity to all 189k games:

```
loss = τ · log Σᵢ exp(cos(x, eᵢ)/τ)  +  λ · relu(c_min − ‖Σ wⱼ aⱼ‖)
```

The first term pushes the point away from every game, including its own ingredients. The second is a **coherence** constraint. Averaging unrelated games cancels what is specific about each one: a blend of a subway sim, a JRPG and a card game collapses into a generic "indie game" vector, which looks far from everything only because real games are specific. The length of the un-normalised blend measures this cancellation, so blends must keep it above `--coherence` (default 0.82), and their ingredients must be related but different (pairwise cosine 0.3–0.8). In the map, the search is limited to the games you picked, but novelty is always measured against all of Steam.

**5. Decoding, closed loop.** To turn a point back into a game, the AI gets a brief: the blend recipe, the tags the probe reads off the point, and the closest existing games, which its idea has to differ from. It writes a store page, which is embedded and measured:

| measure | meaning |
|---|---|
| `target_sim` | cosine between the draft and the void it was aimed at |
| `nearest_sim` | cosine to the closest real Steam game |
| **landed** | `target_sim > nearest_sim`: the draft sits closer to the void than to any existing game |

The measurements go back to the AI for another round, and the best draft is kept, preferring drafts that landed.

**6. Ranking.** Ideas are ranked by **novelty = 1 − nearest_sim** of the idea's own page, plus its **isolation**: the share of real Steam games whose own nearest twin is closer than the idea's. `calibrate` measures how far an AI *rewrite* of an existing game lands from the original: the noise floor from writing style alone, shown on every idea's histogram.

### Steam data

The installer downloads a snapshot of the finished game space from this project's [GitHub release](https://github.com/Obsolete-Robot/IdeaPlotter/releases): embeddings, neighbour tables, clusters, tag probe and map layouts, plus each game's name, tags and review count. Your computer only embeds the drafts the AI writes.

Store descriptions are looked up from Steam's public store API as they are needed: opening a game (one request), writing an idea (one request for the dozen or so games its brief quotes; if Steam is unreachable, the brief uses titles and tags), **Search Steam** / **add to map**, and `calibrate`. Lookups go out one at a time, at most one per second, and each game is cached in `data/steam_cache/` for 30 days.

**Building the data yourself** crawls the whole Steam catalog (about 195 API calls) and embeds 189k games (about 1.5 hours on an RTX 2080 Ti), so it is meant for publishing a new snapshot. With a full build, the app reads all store text from your own data:

```bash
.venv/bin/python -m ideaplotter fetch    # crawl the catalog into data/raw/
.venv/bin/python -m ideaplotter build    # clean, embed, analyse, lay out the map
.venv/bin/python -m ideaplotter export   # optional: pack out/snapshot/ for publishing as a release
```

To publish a new snapshot, attach `out/snapshot/*` to a GitHub release and set `SNAPSHOT_TAG` in `ideaplotter/config.py` to its tag. `download` checks every file against the release's `manifest.json`; `IDEAPLOTTER_SNAPSHOT_URL` points it somewhere else.

### Running it on a server

`serve` can host the map for other people, as on [obsoleterobot.com/ideaplotter](https://obsoleterobot.com/ideaplotter/):

```bash
.venv/bin/python -m ideaplotter serve --host 0.0.0.0 --port 8765 --base-path /ideaplotter \
    --daily-limit 40 --visitor-limit 5 --steam-daily-limit 500
```

- `--base-path` serves the app under a path on a shared domain.
- `--daily-limit` and `--visitor-limit` cap the ideas written per day (in total and per visitor) and write one at a time, which bounds the AI bill. Visitors are identified by Cloudflare's `CF-Connecting-IP` header when present, and stored hashed.
- `--steam-daily-limit` caps the Steam lookups visitors can trigger.

The server log prints each idea's title and, on OpenRouter, what it cost.

### Notes and limits

- The map is a projection. Games far apart on it are dissimilar, but nearby dots can hide high-dimensional gaps, which is why voids are searched in 1024-d.
- Novelty measures distance between *store pages*: how a game is described (theme, mechanics, tags). How good and buildable an idea is gets its own score, the AI's viability judgement.
- Adult-only games (by tag) stay on the map, so the space they occupy counts as taken; blends are made from the other games.
- `IDEAPLOTTER_SPACE=<dir>` points every command at a different data directory.

## License

IdeaPlotter's source code is available under the [MIT License](LICENSE).
Third-party dependencies, model weights, and Steam content are not covered by this license and remain subject to their respective licenses and terms.
