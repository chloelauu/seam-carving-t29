import numpy as np
import pytest
from seamcarve.core import graph_cut_seam, dp_seam, seam_cost, carve


@pytest.mark.parametrize('shape', [(1, 1200), (2, 1200), (1200, 1)])
def test_graph_long_paths_do_not_recurse(shape):
    energy = np.ones(shape)
    seam = graph_cut_seam(energy)
    assert seam_cost(energy, seam) == shape[0]


@pytest.mark.parametrize('method', ['Dynamic programming', 'Greedy', 'Graph cut (max-flow)'])
def test_height_first_matches_transposed_width_first(method):
    image = np.random.default_rng(44).integers(0, 256, (9, 8, 3), dtype=np.uint8)
    a, _ = carve(image, 6, 7, method, 'Height first')
    b, _ = carve(np.swapaxes(image, 0, 1), 7, 6, method, 'Width first')
    np.testing.assert_array_equal(a, np.swapaxes(b, 0, 1))


def test_graph_last_column_and_zero_energy():
    energy = np.array([[8., 7., 0.], [9., 5., 0.], [5., 8., 0.]])
    # Finite variant can remove the zero-energy last column.
    assert graph_cut_seam(energy, last_column='finite').tolist() == [2, 2, 2]
    # Brief's graph cannot: best seam over columns 0..1 is 7 + 5 + 5 = 17.
    assert graph_cut_seam(energy).tolist() == [1, 1, 0]
    assert seam_cost(energy, dp_seam(energy)) == 0
    assert seam_cost(np.zeros((3, 4)), graph_cut_seam(np.zeros((3, 4)))) == 0
