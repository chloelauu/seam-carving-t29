# Seam Carving Lab — Assignment 2

A standalone learning/demo application for dynamic programming, greedy seam selection, and exact min-cut/max-flow seam selection. **No image-library seam carving functions are used.**

## Install and run (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Choose an input image, algorithm, preprocessing limit, width/height targets, and seam order. Press **Carve image**. Tabs show before/after results, a slider displaying **every seam immediately before it is removed**, a projection of **all removed seams on the original image**, input Sobel energy, timings and downloadable output / step JSON. Width and height controls are interactive; reductions only (no enlargement).

**Graph cut safeguard:** the pure-Python Dinic solver limits interactive images to 3600 pixels. Lower the preprocessing limit for large images. The three methods use precisely the same Sobel energy implementation (edge-replicated borders) and vertical/horizontal transposition. Tie-breaking can differ when multiple seams are equally optimal.

## Tests

```powershell
pip install -r requirements-ci.txt
ruff check seamcarve tests app.py
python -m pytest
```

Tests verify DP and graph cut against brute-force enumeration on 1×1, single-column, rectangular, and random energy maps; greedy counterexample; seam adjacency; vertical/horizontal removals; pixel provenance; and invalid input.

## CI/CD

`.github/workflows/ci-cd.yml` runs tests and static checks on pull requests/pushes to `main`. **Only if tests pass**, pushes to `main` (or manual dispatch on main) build and publish a Docker image to GitHub Container Registry as `ghcr.io/<owner>/<repository>:latest` and a commit-hash tag. Docker publishing **will not happen until the workflow is committed and run on your own GitHub repository**. Repo settings must permit Actions to publish packages (`packages:write` is declared in the workflow).

To run a published container:

```bash
docker run --rm -p 8501:8501 ghcr.io/OWNER/REPO:latest
```

Then open http://localhost:8501. To host the Streamlit UI for public users, configure a runtime separately (e.g., Streamlit Community Cloud linked to the same GitHub repo); this is **not** automatically performed by the container-publishing workflow.

## Algorithm contracts

- `seamcarve.core.energy_map`: backward Sobel gradient magnitude `abs(Gx)+abs(Gy)`.
- `dp_seam`: globally minimal cumulative energy, O(HW) time and O(HW) parent array (current and previous cost rows only).
- `greedy_seam`: min top row, then min of three valid successors; the mathematical selection loop takes O(W+H). The public function first validates the entire energy array, so its implemented call takes O(HW) time and O(HW) temporary validation space. It is not globally optimal.
- `graph_cut_seam`: original graph construction using source-anchored left side, finite per-pixel rightward/terminal edges, infinite backward and adjacent-row diagonal-left edges. Maximum flow uses Dinic. Nodes/edges Θ(HW). Complexity: general Dinic upper bound O(V²E), space Θ(V+E). Terminal edge permits a seam in the last column.
- `carve`: iteratively recomputes energy, removes one seam at a time, maps removed pixels back to the **original image** for full-history visualization.

## Reproducibility

From the project root, install the optional plotting dependencies and run:

```powershell
pip install -r requirements-benchmark.txt
python -m scripts.evaluate
```

This reproduces the synthetic and three real-photo benchmarks, with one excluded warm-up and five measured runs, first-map/first-solver timings, full resize timings, source hashes, environment details, and figures. Data and image credits are in `assets/photos/sources.json`. Full history is enabled, so timings include frame copies and original-pixel provenance. `assets/evidence/evaluation.json` stores all measurements. The older figure generator can be run with `python -m scripts.benchmark`; it uses five consecutive trials without a warm-up and is retained for Figures 1 and 2.

The revised package passes 36 tests (20 original core cases, seven core regressions, nine Streamlit interaction cases) and Ruff locally. GitHub-hosted Actions and container publication still require a run in your team's repository. Record the final Git commit and demonstrate the app in the presentation.

The app stores results in Streamlit session state, preserves them during history navigation and downloads, and clears them when the input or resize settings change. One-seam and one-pixel dimensions do not create zero-range sliders. The flow solver uses an explicit DFS stack to support long thin images without Python recursion failures. Full-frame history uses O(KHW) upper-bound storage across K removals; use `record=False` in core-only workflows when it is unnecessary.

## Academic-integrity check

The code and report were AI-assisted. Review and understand them, record the main AI prompts in your declaration, and independently confirm all final screenshots, timings, repository URLs and CI/CD run status. The assignment includes a closed-book team quiz on your submitted work.
