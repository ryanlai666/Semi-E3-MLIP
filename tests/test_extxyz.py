import io
import numpy as np
import pytest
from semi_mlip.extxyz import read_extxyz


def test_columns_follow_schema_not_fixed_offsets():
    text = '1\nLattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:forces:R:3:pos:R:3 energy=-2\nSi 1 2 3 4 5 6\n'
    meta, cols = next(read_extxyz(io.StringIO(text)))
    assert meta['energy'] == '-2'
    np.testing.assert_array_equal(cols['pos'], [[4, 5, 6]])
    np.testing.assert_array_equal(cols['forces'], [[1, 2, 3]])


@pytest.mark.parametrize('row', ['Si 0 1', 'Si nan 1 2'])
def test_bad_numeric_data_rejected(row):
    with pytest.raises(ValueError):
        list(read_extxyz(io.StringIO('1\nProperties=species:S:1:pos:R:3\n'+row+'\n')))
