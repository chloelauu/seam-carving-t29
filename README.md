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

`.github/workflows/ci-cd.yml` runs tests and static checks on pushes to `main` and on pull requests into `main`, and checks that `assets/evidence/source_hashes.json` matches the committed code. **Only if tests pass**, a push to `main` (or a manual dispatch on `main`) builds the Docker image and pushes it under the commit-hash tag only, then pulls that image back from GitHub Container Registry, starts it, waits for Streamlit's health endpoint, runs the app script inside the container with `AppTest`, and only after all of that copies the tag to `latest` — so `latest` never names an image that was merely built. Repo settings must permit Actions to publish packages (`packages:write` is declared in the workflow).

To run a published container:

```bash
docker run --rm -p 8501:8501 ghcr.io/OWNER/REPO:latest
```

Then open http://localhost:8501. To host the Streamlit UI for public users, configure a runtime separately (e.g., Streamlit Community Cloud linked to the same GitHub repo); this is **not** automatically performed by the container-publishing workflow.

## Algorithm contracts

- `seamcarve.core.energy_map`: backward Sobel gradient magnitude `abs(Gx)+abs(Gy)`.
- `dp_seam`: globally minimal cumulative energy, O(HW) time and O(HW) parent array (current and previous cost rows only).
- `greedy_seam`: min top row, then min of three valid successors; the mathematical selection loop takes O(W+H). The public function first validates the entire energy array, so its implemented call takes O(HW) time and O(HW) temporary validation space. It is not globally optimal.
- `graph_cut_seam`: graph construction from the brief using a source-anchored left side, a finite per-pixel rightward edge, an **infinite** terminal edge from every last-column pixel to the sink (so a cut can never remove a last-column pixel), and infinite backward and adjacent-row diagonal-left edges. `last_column='finite'` swaps that terminal edge for a finite one of capacity `e(i, W-1)`, which makes the sink act as an extra column and the cut equal the unrestricted DP optimum; the brief's infinite construction is the default and produces every graph-cut result in the report. Maximum flow uses Dinic. Nodes/edges Θ(HW). Complexity: general Dinic upper bound O(V²E), space Θ(V+E).
- `carve`: iteratively recomputes energy, removes one seam at a time, maps removed pixels back to the **original image** for full-history visualization.

## Reproducibility

From the project root, install the optional plotting dependencies and run:

```powershell
pip install -r requirements-benchmark.txt
python -m scripts.evaluate
```

This reproduces the synthetic and three real-photo benchmarks, with one excluded warm-up and five measured runs, first-map/first-solver timings, full resize timings, source hashes, environment details, and figures. Data and image credits are in `assets/photos/sources.json`. Full history is enabled, so timings include frame copies and original-pixel provenance. `assets/evidence/evaluation.json` stores all measurements. The older figure generator can be run with `python -m scripts.benchmark`; it uses five consecutive trials without a warm-up and is retained for Figures 1 and 2.

The revised package passes 37 tests (21 original core cases, seven core regressions, nine Streamlit interaction cases) and Ruff locally. The workflow in `.github/workflows/ci-cd.yml` runs the same static checks and tests on every push to `main` and every pull request into `main`, publishes a container to GitHub Container Registry from `main` only after they pass, and pushes the `latest` tag only once the published image has been started and its script has run without an exception. That verification executes on a push to `main` and has not yet run: as of 17 November 2026 the only hosted run in the repository's history is the one for the base commit. `assets/evidence/source_hashes.json` records the SHA-256 of each covered file's LF-normalised content, is regenerated with `python -m scripts.hashes`, and is checked by the tests job. Record the final Git commit and demonstrate the app in the presentation.

The app stores results in Streamlit session state, preserves them during history navigation and downloads, and clears them when the input or resize settings change. One-seam and one-pixel dimensions do not create zero-range sliders. The flow solver uses an explicit DFS stack to support long thin images without Python recursion failures. Full-frame history uses O(KHW) upper-bound storage across K removals; use `record=False` in core-only workflows when it is unnecessary.

## Academic-integrity check

The code and report were AI-assisted. Review and understand them, record the main AI prompts in your declaration, and independently confirm all final screenshots, timings, repository URLs and CI/CD run status. The assignment includes a closed-book team quiz on your submitted work.
