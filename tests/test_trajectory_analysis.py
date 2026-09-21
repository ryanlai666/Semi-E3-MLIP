import numpy as np
from scripts.compare_aimd import displacement, pair_histogram

def test_periodic_displacement_unwrap_and_pair_counts():
    rows=[{'positions':[[9.9,0,0],[5.,0,0]],'cell':(np.eye(3)*10).tolist()},
          {'positions':[[.1,0,0],[4.8,0,0]],'cell':(np.eye(3)*10).tolist()}]
    np.testing.assert_allclose(displacement(rows),[0,.04],atol=1e-12)
    hist=pair_histogram(rows,np.linspace(0,5,11))
    assert abs(np.sum(hist)*.5-1)<1e-12


def test_skew_minimum_image_matches_bruteforce_and_lattice_translation():
    import itertools
    from scripts.compare_aimd import minimum_image
    cell=np.array([[10.,0,0],[9.,2.,0],[0,0,10.]])
    delta=np.array([[.49,.49,0],[-.48,.45,.2]])@cell
    shifts=np.array(list(itertools.product(range(-5,6),range(-5,6),range(-2,3))))
    candidates=delta[:,None,:]+shifts@cell
    expected=np.min(np.sum(candidates**2,axis=-1),axis=1)
    actual=minimum_image(delta,cell)
    np.testing.assert_allclose(np.sum(actual**2,axis=1),expected,atol=1e-12)
    translated=minimum_image(delta+np.array([3,-2,1])@cell,cell)
    np.testing.assert_allclose(translated,actual,atol=1e-12)
    assert np.linalg.norm(actual[0])<1.1  # Rounding fractional coordinates alone fails.


def test_skew_unwrap_and_pair_histogram():
    cell=np.array([[10.,0,0],[9.,2.,0],[0,0,10.]])
    initial=np.array([[.95,.95,.2],[.4,.4,.2]])@cell
    motion=np.array([[.1,.1,0],[-.1,-.1,0]])
    wrapped=((initial+motion)@np.linalg.inv(cell)%1)@cell
    rows=[{'positions':x.tolist(),'cell':cell.tolist()} for x in (initial,wrapped)]
    np.testing.assert_allclose(displacement(rows),[0,.02],atol=1e-12)
    pair={'positions':[[0.,0,0],*[((np.array([.49,.49,0])@cell).tolist())]],'cell':cell.tolist()}
    np.testing.assert_allclose(pair_histogram([pair],np.array([0.,1.,2.,3.])),[0,1,0])


def test_minimum_image_preserves_nonperiodic_direction():
    from scripts.compare_aimd import minimum_image
    cell=np.diag([10.,10.,10.])
    np.testing.assert_allclose(minimum_image([[11.,12.,13.]],cell,[True,False,True]),[[1.,12.,3.]])
