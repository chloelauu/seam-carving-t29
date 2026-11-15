import numpy as np
import pytest
from itertools import product
from seamcarve.core import (dp_seam, greedy_seam, graph_cut_seam, seam_cost,
                            remove_vertical, energy_map, carve, all_seams_overlay)

COUNTEREXAMPLE = np.array([[1,2,50,50],[8,9,1,50],[50,49,1,50],[50,49,1,50]],dtype=float)

def brute_cost(e):
    h,w=e.shape
    best=float('inf')
    for seam in product(range(w),repeat=h):
        if all(abs(a-b)<=1 for a,b in zip(seam,seam[1:])):
            best=min(best,sum(e[r,c] for r,c in enumerate(seam)))
    return best

def test_counterexample():
    d=dp_seam(COUNTEREXAMPLE)
    g=greedy_seam(COUNTEREXAMPLE)
    f=graph_cut_seam(COUNTEREXAMPLE)
    assert d.tolist()==[1,2,2,2]
    assert g.tolist()==[0,0,1,2]
    assert np.isclose(seam_cost(COUNTEREXAMPLE,d),5)
    assert np.isclose(seam_cost(COUNTEREXAMPLE,g),59)
    assert np.isclose(seam_cost(COUNTEREXAMPLE,f),5)

@pytest.mark.parametrize('shape',[(1,1),(1,4),(4,1),(2,5),(5,2),(3,4),(4,4)])
def test_optimality_against_brute_force(shape):
    rng=np.random.default_rng(sum(shape)*100+shape[1])
    for _ in range(8):
        e=rng.integers(0,30,size=shape).astype(float)
        optimal=brute_cost(e)
        assert seam_cost(e,dp_seam(e))==optimal
        # Finite last-column variant: equals the unrestricted optimum.
        assert seam_cost(e,graph_cut_seam(e,last_column='finite'))==optimal
        # Brief's graph (default): optimum over every column except the last.
        blocked=e.copy()
        if shape[1]>1:
            blocked[:,-1]=1e9
        assert seam_cost(e,graph_cut_seam(e))==brute_cost(blocked)


def test_brief_graph_never_uses_last_column():
    rng=np.random.default_rng(31)
    for _ in range(40):
        h,w=rng.integers(1,8),rng.integers(2,8)
        e=rng.random((h,w))*50
        assert graph_cut_seam(e).max()<=w-2

@pytest.mark.parametrize('shape',[(2,3),(3,7),(9,9)])
def test_methods_return_connected_valid_seams(shape):
    e=np.random.default_rng(2042).random(shape)
    for solver in (dp_seam,greedy_seam,graph_cut_seam):
        s=solver(e)
        assert len(s)==shape[0]
        assert np.all(s>=0) and np.all(s<shape[1])
        assert np.all(np.abs(np.diff(s))<=1)

def test_removal_preserves_order():
    a=np.arange(20).reshape(4,5)
    seam=np.array([1,1,2,2])
    result=remove_vertical(a,seam)
    assert result.shape==(4,4)
    for r in range(4):
        assert result[r].tolist()==np.delete(a[r],seam[r]).tolist()

@pytest.mark.parametrize('method',['Dynamic programming','Greedy','Graph cut (max-flow)'])
def test_vertical_horizontal_history_and_provenance(method):
    img=np.random.default_rng(12).integers(0,256,size=(8,9,3),dtype=np.uint8)
    result,steps=carve(img,7,6,method,'Width first')
    assert result.shape==(6,7,3)
    assert len(steps)==4
    assert [s.orientation for s in steps]==['vertical','vertical','horizontal','horizontal']
    points=[tuple(x) for step in steps for x in step.removed_original_coords]
    assert len(points)==len(set(points))
    assert all(0<=r<8 and 0<=c<9 for r,c in points)
    assert all_seams_overlay(img,steps).shape==img.shape

def test_single_dimension_and_uniform_image():
    img=np.full((1,8,3),100,dtype=np.uint8)
    assert np.count_nonzero(energy_map(img))==0
    resized,steps=carve(img,1,1,'Dynamic programming')
    assert resized.shape==(1,1,3)
    assert len(steps)==7
    with pytest.raises(ValueError): carve(img,0,1)
    with pytest.raises(ValueError): carve(img,9,1)

@pytest.mark.parametrize('bad', [np.array([[np.nan]]),np.array([[-1]]),np.array([])])
def test_invalid_energies(bad):
    with pytest.raises(ValueError): dp_seam(bad)

def test_graph_cut_limit():
    with pytest.raises(ValueError,match='limit'):
        graph_cut_seam(np.ones((61,60)))
