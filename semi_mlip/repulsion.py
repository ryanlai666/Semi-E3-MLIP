"""Experimental screened nuclear core in eV/Angstrom; not a full force field.

Universal screening coefficients follow the LAMMPS ZBL documentation.
Our multiplicative quintic switch differs from LAMMPS's additive switch.
"""
import torch

def switched_zbl(distance, zi, zj, inner=0.8, outer=1.5):
    if not 0 < inner < outer:
        raise ValueError('Expected 0 < inner < outer')
    zi,zj=zi.to(distance),zj.to(distance)
    # Graph construction rejects overlaps. Do not silently clip distances/forces.
    if bool((distance <= 0).any()):raise ValueError('ZBL requires positive distances')
    screening_length=0.46850/(zi.pow(.23)+zj.pow(.23))
    x=distance/screening_length
    phi=(.18175*torch.exp(-3.19980*x)+.50986*torch.exp(-.94229*x)
         +.28022*torch.exp(-.40290*x)+.02817*torch.exp(-.20162*x))
    u=((distance-inner)/(outer-inner)).clamp(0,1)
    switch=(1-u).pow(3)*(1+3*u+6*u.square())
    return 14.3996454784255*zi*zj*phi*switch/distance
