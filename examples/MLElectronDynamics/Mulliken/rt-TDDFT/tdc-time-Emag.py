import numpy as np 
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter
from ase.units import Hartree, Bohr
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.laser import GaussianPulse
'''
# Temporal shape of the time-dependent potential
Emag = 1.2e-05
E_hat = np.array([0.486833, 0.873484, 0.004399])
pulse = GaussianPulse(Emag, 2.5e3, 3.8, 0.75, 'sin')
ext = ConstantElectricField(Hartree / Bohr, E_hat)
td_potential = {'ext': ext, 'laser': pulse}
pulse.write("pulse.dat", np.arange(0, 30e3, 10.0))
'''
# Parallelzation settings
parallel = {'sl_auto': True, 'use_elpa':True, 'augment_grids': True}
# Time propagation
td_calc = LCAOTDDFT('tdc3.gpw', parallel=parallel, txt='tdc4.out')
DipoleMomentWriter(td_calc, 'dm.dat')
WaveFunctionWriter(td_calc, 'wf.ulm', split=True)

# propagate
td_calc.propagate(20, 400)

# Save the state for restarting later
td_calc.write('tdc4.gpw', mode='all')
