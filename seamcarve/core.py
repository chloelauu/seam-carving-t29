"""Original seam-carving algorithms; no content-aware resizing library is used."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import colorsys
import numpy as np
from PIL import Image

MAX_GRAPH_PIXELS = 3600  # practical UI safeguard for a pure-Python flow solver


def energy_map(image: np.ndarray) -> np.ndarray:
    """Sobel L1 gradient on luminance with edge-replicated boundary pixels."""
    im = np.asarray(image, dtype=np.float64)
    if im.ndim == 3:
        if im.shape[2] < 3:
            raise ValueError('Expected RGB/RGBA pixels or a grayscale array')
        gray = 0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2]
    elif im.ndim == 2:
        gray = im
    else:
        raise ValueError('Expected 2D grayscale or 3D RGB image array')
    if not gray.size:
        raise ValueError('Image must be nonempty')
    p = np.pad(gray, ((1, 1), (1, 1)), mode='edge')
    gx = (-p[:-2, :-2] + p[:-2, 2:] - 2*p[1:-1, :-2] + 2*p[1:-1, 2:]
          - p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:])
    return np.abs(gx) + np.abs(gy)


def validate_energy(e):
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 2 or 0 in e.shape or not np.isfinite(e).all() or (e < 0).any():
        raise ValueError('Energy must be a nonempty 2D array of finite nonnegative values')
    return e


def seam_cost(energy, seam) -> float:
    e = validate_energy(energy)
    s = np.asarray(seam, dtype=np.int64)
    if s.shape != (e.shape[0],) or np.any(s < 0) or np.any(s >= e.shape[1]):
        raise ValueError('Invalid seam shape or column index')
    if np.any(np.abs(np.diff(s)) > 1):
        raise ValueError('Seam is not 8-connected')
    return float(np.sum(e[np.arange(len(s)), s]))


def dp_seam(energy: np.ndarray) -> np.ndarray:
    """Globally optimal additive backward-energy seam; leftmost ties."""
    e = validate_energy(energy)
    h, w = e.shape
    costs = e[0].copy()
    parents = np.empty((h, w), dtype=np.int32)
    parents[0] = -1
    for r in range(1, h):
        left = np.r_[np.inf, costs[:-1]]
        center = costs
        right = np.r_[costs[1:], np.inf]
        options = np.stack((left, center, right))
        choices = np.argmin(options, axis=0)   # if tied, leftmost
        parents[r] = np.arange(w) + choices - 1
        costs = e[r] + np.min(options, axis=0)
    s = np.empty(h, dtype=np.int32)
    s[-1] = int(np.argmin(costs))
    for r in range(h - 1, 0, -1):
        s[r-1] = parents[r, s[r]]
    return s


def greedy_seam(energy: np.ndarray) -> np.ndarray:
    """Select minimum top-row energy, then minimum legal local neighbour."""
    e = validate_energy(energy)
    h, w = e.shape
    s = np.empty(h, dtype=np.int32)
    s[0] = int(np.argmin(e[0]))
    for r in range(1, h):
        lo, hi = max(0, int(s[r-1])-1), min(w, int(s[r-1])+2)
        s[r] = lo + int(np.argmin(e[r, lo:hi]))
    return s


class _Dinic:
    """Dinic maximum flow on an adjacency list.

    Each edge is stored as [to, index_of_reverse_edge, remaining_capacity].
    Every forward edge has a paired reverse edge that starts at capacity 0;
    pushing flow along an edge gives the same amount of capacity to its reverse,
    which is what lets later paths "undo" earlier choices (the residual graph).
    """

    def __init__(self, n):
        self.adj = [[] for _ in range(n)]
        self.n = n

    def add(self, u, v, capacity):
        fwd = [v, len(self.adj[v]), float(capacity)]
        rev = [u, len(self.adj[u]), 0.0]
        self.adj[u].append(fwd)
        self.adj[v].append(rev)

    def maxflow(self, source, sink):
        eps = 1e-9
        total = 0.0
        while True:
            # Phase 1 - BFS builds the level graph: level[v] = shortest number of
            # edges with spare capacity from the source to v.
            levels = [-1]*self.n
            levels[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v, _, cap in self.adj[u]:
                    if cap > eps and levels[v] < 0:
                        levels[v] = levels[u]+1
                        q.append(v)
            if levels[sink] < 0:
                break  # sink unreachable: current flow is maximum
            # pos[u] = next edge of u still worth trying in this phase
            # ("current-arc" pointer), so dead edges are never re-scanned.
            pos = [0]*self.n

            def send():
                # Phase 2 - find ONE source->sink path that only moves to the next
                # level, and push its bottleneck capacity. An explicit stack is used
                # instead of recursion so long thin images cannot overflow Python's stack.
                nodes = [source]
                path = []
                limits = [float('inf')]
                while nodes:
                    u = nodes[-1]
                    if u == sink:
                        pushed = limits[-1]
                        for parent, edge_index in path:
                            edge = self.adj[parent][edge_index]
                            edge[2] -= pushed                       # use forward capacity
                            self.adj[edge[0]][edge[1]][2] += pushed  # add reverse capacity
                        return pushed
                    while pos[u] < len(self.adj[u]):
                        v, _, cap = self.adj[u][pos[u]]
                        if cap > eps and levels[v] == levels[u]+1:
                            break
                        pos[u] += 1
                    if pos[u] == len(self.adj[u]):
                        # dead end: retreat and skip the edge that led here
                        nodes.pop()
                        limits.pop()
                        if path:
                            parent, _ = path.pop()
                            pos[parent] += 1
                    else:
                        v, _, cap = self.adj[u][pos[u]]
                        path.append((u, pos[u]))
                        nodes.append(v)
                        limits.append(min(limits[-1], cap))
                return 0.0
            while True:  # repeat until the level graph is blocked
                pushed = send()
                if pushed <= eps:
                    break
                total += pushed
        return total

    def reachable(self, source):
        """Vertices reachable from the source through edges with spare capacity.
        After max flow this is the source side S of a minimum cut."""
        seen = {source}
        q = deque([source])
        while q:
            u = q.popleft()
            for v, _, cap in self.adj[u]:
                if cap > 1e-9 and v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen


def graph_cut_seam(energy: np.ndarray, max_pixels: int | None = MAX_GRAPH_PIXELS,
                   last_column: str = 'infinite') -> np.ndarray:
    """Minimum-energy vertical seam as an s-t minimum cut (brief item 4).

    Pixel (r,c) ends on the source side iff c <= seam[r]. Edges:
      source -> (r,0)            INF       first column always on the source side
      (r,c) -> (r,c+1)           e[r,c]    the only cuttable edges: cutting = removing (r,c)
      (r,c+1) -> (r,c)           INF       source side of each row is a prefix (one cut per row)
      (r,c) -> (r-1,c-1),(r+1,c-1) INF     cut positions in adjacent rows differ by <= 1
      (r,w-1) -> sink            see last_column

    last_column='infinite' (default) is exactly the brief's construction: the last
    column can never be cut, so the seam is optimal over columns 0..w-2.
    last_column='finite' gives that edge capacity e[r,w-1], so the sink acts as an
    extra column and the result equals the unrestricted DP optimum.
    """
    if last_column not in ('infinite', 'finite'):
        raise ValueError("last_column must be 'infinite' or 'finite'")
    e = validate_energy(energy)
    h, w = e.shape
    if max_pixels is not None and h*w > max_pixels:
        raise ValueError(f'Graph-cut safety limit: {max_pixels} pixels; downsample the image')
    if w == 1:
        # Only one possible seam. With the brief's INF edges a 1-column graph has
        # no finite cut at all, so it is handled directly.
        return np.zeros(h, dtype=np.int32)
    source, sink = h*w, h*w+1
    g = _Dinic(h*w+2)
    inf = float(np.sum(e) + 1.0)  # larger than any finite cut, so INF edges are never cut
    idx = lambda r, c: r*w+c
    for r in range(h):
        g.add(source, idx(r, 0), inf)
        for c in range(w):
            if c == w-1:
                g.add(idx(r, c), sink, inf if last_column == 'infinite' else e[r, c])
            else:
                g.add(idx(r,c), idx(r,c+1), e[r,c])
                g.add(idx(r,c+1), idx(r,c), inf)
            if c > 0:
                if r > 0:
                    g.add(idx(r,c), idx(r-1,c-1), inf)
                if r < h-1:
                    g.add(idx(r,c), idx(r+1,c-1), inf)
    best = g.maxflow(source, sink)
    reach = g.reachable(source)
    # In each row the seam pixel is the rightmost pixel on the source side.
    seam = np.array([max(c for c in range(w) if idx(r,c) in reach) for r in range(h)], dtype=np.int32)
    # Max-flow min-cut theorem: the flow value must equal the cut (= seam) cost.
    if not np.isclose(seam_cost(e,seam), best, rtol=1e-7, atol=1e-6):
        raise AssertionError('Cut capacity does not equal seam energy')
    return seam

SOLVERS = {'Dynamic programming': dp_seam, 'Greedy': greedy_seam, 'Graph cut (max-flow)': graph_cut_seam}


def remove_vertical(image, seam):
    arr = np.asarray(image)
    if arr.ndim < 2 or arr.shape[1] <= 1:
        raise ValueError('Cannot remove seam from image with width <= 1')
    h,w = arr.shape[:2]
    s = np.asarray(seam)
    if s.shape != (h,) or np.any(s<0) or np.any(s>=w) or np.any(np.abs(np.diff(s))>1):
        raise ValueError('Invalid vertical seam')
    mask = np.ones((h,w), dtype=bool)
    mask[np.arange(h),s] = False
    return arr[mask].reshape((h,w-1)+arr.shape[2:])


def overlay_seam(image: np.ndarray, seam: np.ndarray, orientation='vertical') -> np.ndarray:
    out = np.array(image, dtype=np.uint8, copy=True)
    if orientation == 'vertical':
        out[np.arange(len(seam)), seam, :] = [255, 30, 75]
    else:
        out[seam, np.arange(len(seam)), :] = [255, 30, 75]
    return out


@dataclass
class Step:
    number: int
    orientation: str
    seam: np.ndarray
    before: np.ndarray
    cost: float
    removed_original_coords: np.ndarray  # shape (length, 2), (row, col)


def carve(image: np.ndarray, target_width: int, target_height: int,
          algorithm='Dynamic programming', order='Width first', record=True):
    original = np.asarray(image, dtype=np.uint8)
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('Input should be an RGB image array')
    h, w = original.shape[:2]
    if not (1 <= target_width <= w and 1 <= target_height <= h):
        raise ValueError('Targets must be positive and cannot exceed input dimensions')
    if algorithm not in SOLVERS:
        raise ValueError('Unrecognised algorithm')
    if order not in ('Width first','Height first'):
        raise ValueError('Unrecognised seam-removal order')
    if algorithm == 'Graph cut (max-flow)' and h*w > MAX_GRAPH_PIXELS and (target_width<w or target_height<h):
        raise ValueError(f'Graph cut is limited to {MAX_GRAPH_PIXELS} input pixels for interactive use')
    work = original.copy()
    ys, xs = np.indices((h,w), dtype=np.int32)
    steps=[]
    dirs = [('vertical',w-target_width),('horizontal',h-target_height)]
    if order == 'Height first':
        dirs.reverse()
    for direction, count in dirs:
        for _ in range(count):
            transposed = direction == 'horizontal'
            oriented = np.swapaxes(work,0,1) if transposed else work
            coords_y = ys.T if transposed else ys
            coords_x = xs.T if transposed else xs
            e = energy_map(oriented)
            s = SOLVERS[algorithm](e)
            cost = seam_cost(e,s)
            positions = np.column_stack((coords_y[np.arange(len(s)),s], coords_x[np.arange(len(s)),s]))
            if record:
                steps.append(Step(len(steps)+1,direction,s.copy(),work.copy(),cost,positions))
            oriented_new = remove_vertical(oriented,s)
            new_y = remove_vertical(coords_y,s)
            new_x = remove_vertical(coords_x,s)
            work = np.swapaxes(oriented_new,0,1) if transposed else oriented_new
            ys = new_y.T if transposed else new_y
            xs = new_x.T if transposed else new_x
    return work, steps


def all_seams_overlay(original: np.ndarray, steps: list[Step]) -> np.ndarray:
    out = np.asarray(original, dtype=np.uint8).copy()
    count=max(1,len(steps))
    for k, step in enumerate(steps):
        # Distinct hue across the whole carving history
        rgb = np.asarray(colorsys.hsv_to_rgb(k/count*0.82,0.94,1.0))*255
        rr,cc = step.removed_original_coords.T
        out[rr,cc] = rgb.astype(np.uint8)
    return out


def resize_max_side(pil_image: Image.Image, limit: int) -> np.ndarray:
    im = pil_image.convert('RGB')
    im.thumbnail((limit,limit),Image.Resampling.LANCZOS)
    return np.asarray(im)
