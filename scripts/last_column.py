"""Isolate the effect of the brief's graph (infinite last-column edges).

Run from the repo root:  python -m scripts.last_column

At every step of a DP carve, solve the SAME energy map with DP and with the
brief's graph cut, and count the steps where graph cut is dearer and the steps
where DP's best seam uses the last column. Uses the same inputs and targets as
scripts/evaluate.py.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from seamcarve.core import dp_seam, energy_map, graph_cut_seam, remove_vertical, resize_max_side, seam_cost

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [('Synthetic', 'assets/example.png'), ('Astronaut', 'assets/photos/astronaut.png'),
          ('Coffee', 'assets/photos/coffee.png'), ('Rocket', 'assets/photos/rocket.jpg')]


def main():
    total_steps = total_dearer = 0
    for name, rel in INPUTS:
        pil = Image.open(ROOT / rel).convert('RGB')
        image = np.asarray(pil) if name == 'Synthetic' else resize_max_side(pil, 60)
        h, w = image.shape[:2]
        tw, th = (58, 44) if name == 'Synthetic' else (round(w * .75), h)
        work = image.copy()
        steps = dearer = uses_last = both = 0
        for transposed, count in ((False, w - tw), (True, h - th)):
            for _ in range(count):
                oriented = np.swapaxes(work, 0, 1) if transposed else work
                e = energy_map(oriented)
                s_dp = dp_seam(e)
                s_gc = graph_cut_seam(e)  # default = brief's graph
                is_dearer = seam_cost(e, s_gc) > seam_cost(e, s_dp) + 1e-6
                is_last = bool((s_dp == e.shape[1] - 1).any())
                steps += 1
                dearer += is_dearer
                uses_last += is_last
                both += is_dearer and is_last
                oriented = remove_vertical(oriented, s_dp)
                work = np.swapaxes(oriented, 0, 1) if transposed else oriented
        total_steps += steps
        total_dearer += dearer
        print(f'{name}: {steps} steps | graph cut dearer: {dearer} | '
              f'DP seam uses last column: {uses_last} | both: {both}')
    print(f'Overall: graph cut dearer at {total_dearer} of {total_steps} steps')


if __name__ == '__main__':
    main()
