import numpy as np
from scripts.compare_aimd import displacement, pair_histogram

def test_periodic_displacement_unwrap_and_pair_counts():
    rows=[{'positions':[[9.9,0,0],[5.,0,0]],'cell':(np.eye(3)*10).tolist()},
          {'positions':[[.1,0,0],[4.8,0,0]],'cell':(np.eye(3)*10).tolist()}]
    np.testing.assert_allclose(displacement(rows),[0,.04],atol=1e-12)
    hist=pair_histogram(rows,np.linspace(0,5,11))
    assert abs(np.sum(hist)*.5-1)<1e-12
