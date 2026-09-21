"""Fixed periodic-table facts; no geometry- or DFT-dependent input labels."""
import torch

# (period, IUPAC group). Optional weak chemical prior, not an oxidation state.
PERIOD_GROUP = {8: (2,16), 13: (3,13), 14: (3,14), 22: (4,4), 27: (4,9),
                29: (4,11), 40: (5,4), 44: (5,8), 72: (6,4), 73: (6,5), 74: (6,6)}


def descriptor_table():
    table = torch.zeros(119, 4)
    for z, (period, group) in PERIOD_GROUP.items():
        table[z] = torch.tensor([z/100., period/7., group/18., float(3 <= group <= 12)])
    return table
