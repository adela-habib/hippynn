from ase.cluster import Decahedron
from ase.io import read, write
from gpaw import GPAW, FermiDirac, Mixer, PoissonSolver
from gpaw.poisson_moment import MomentCorrectionPoissonSolver
from ase.cluster.cubic import FaceCenteredCubic

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [4, 7, 3]
lc = 4.09
atoms = FaceCenteredCubic('Ag', surfaces, layers, latticeconstant=lc)
write('Ag201.xyz', atoms)
atoms.center(vacuum=10.0)

# I ncrease the accuracy of density for ground state
convergence = {'density': 1e-12}

# Use occupation smearing and weak mixer to facilitate convergence
occupations = FermiDirac(25e-3)
mixer = Mixer(0.02, 5, 1.0)

# Parallelzation settings
parallel = {'domain': 16}

# Apply multipole corrections for monopole and dipoles
poissonsolver = MomentCorrectionPoissonSolver(poissonsolver=PoissonSolver(),moment_corrections=1+3) 

# Ground-state calculation
calc = GPAW(mode='lcao', xc='GLLBSC', h=0.3, nbands='nao',
            setups={'Ag': 'my'},
            basis={'Ag': 'GLLBSC.dz', 'default': 'dzp'},
            convergence=convergence, poissonsolver=poissonsolver,
            occupations=occupations, mixer=mixer, parallel=parallel,
            maxiter=2000,
            txt='gs.out')
atoms.calc = calc
atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')
