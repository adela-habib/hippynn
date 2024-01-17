"""
Layers for physical operations
"""
import warnings

import torch
from torch import Tensor

from . import indexers, pairs


class Gradient(torch.nn.Module):
    def __init__(self, sign):
        super().__init__()
        assert sign in (-1, 1), "Sign of gradient must be +1 (gradient) or -1 (force)"
        self.sign = sign

    def forward(self, molecular_energies, positions):
        return self.sign * torch.autograd.grad(molecular_energies.sum(), positions, create_graph=True)[0]


class StressForce(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbc = False

    def forward(self, energy, strain, coordinates, cell):
        total_energy = energy.sum()
        straingrad, grad = torch.autograd.grad(total_energy, [strain, coordinates], create_graph=self.training)
        if self.pbc:
            stress = straingrad / torch.det(cell)
        else:
            stress = straingrad
        return -grad, stress


class Dipole(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.summer = indexers.MolSummer()

    def forward(self, charges: Tensor, positions: Tensor, mol_index: Tensor, n_molecules: int):
        if charges.shape[1] > 1:
            # charges contain multiple targets, so set up broadcasting
            charges = charges.unsqueeze(2)
            positions = positions.unsqueeze(1)

        # shape is (n_atoms, 3, n_targets) in multi-target mode
        # shape is (n_atoms, 3) in single target mode
        dipole_elements = charges * positions
        dipoles = self.summer(dipole_elements, mol_index, n_molecules)
        return dipoles


class Quadrupole(torch.nn.Module):
    """Computes quadrupoles as a flattened (n_molecules,9) array.
    NOTE: Uses normalization sum_a q_a (r_a,i*r_a,j - 1/3 delta_ij r_a^2)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.summer = indexers.MolSummer()

    def forward(self, charges, positions, mol_index, n_molecules):
        # positions shape: (atoms, xyz)
        # charge shape: (atoms,1)
        ri_rj = positions.unsqueeze(1) * positions.unsqueeze(2)
        ri_rj_flat = ri_rj.reshape(-1, 9)  # Flatten to component
        rsq = (positions**2).sum(dim=1).unsqueeze(1)  # unsqueeze over component index
        delta_ij = torch.eye(3, device=rsq.device).flatten().unsqueeze(0)  # unsqueeze over atom index
        quad_elements = charges * (ri_rj_flat - (1 / 3) * (rsq * delta_ij))
        quadrupoles = self.summer(quad_elements, mol_index, n_molecules)
        return quadrupoles


class CoulombEnergy(torch.nn.Module):
    """ Computes the Coulomb Energy of the molecule/configuration. 
    
    Coulomb energies is defined for pairs of atoms. Here, we adopt the 
    convention that the Coulomby energy for a pair of atoms is evenly
    partitioned to both atoms as the 'per-atom energies'. Therefore, the 
    atom energies sum to the molecular energy; similar to the HEnergy. 
    """
    def __init__(self, energy_conversion_factor):
        super().__init__()
        self.register_buffer("energy_conversion_factor", torch.tensor(energy_conversion_factor))
        self.summer = indexers.MolSummer()

    def forward(self, charges, pair_dist, pair_first, pair_second, mol_index, n_molecules):
        voltage_pairs = self.energy_conversion_factor * (charges[pair_second] / pair_dist.unsqueeze(1))
        n_atoms, _ = charges.shape
        voltage_atom = torch.zeros((n_atoms, 1), device=charges.device, dtype=charges.dtype)
        voltage_atom.index_add_(0, pair_first, voltage_pairs)
        coulomb_atoms = 0.5*voltage_atom * charges
        coulomb_molecule = self.summer(coulomb_atoms, mol_index, n_molecules)
        return coulomb_molecule, coulomb_atoms, voltage_atom


class ScreenedCoulombEnergy(CoulombEnergy):
    """ Computes the Coulomb Energy of the molecule/configuration. 
    
    The convention for the atom energies is the same as CoulombEnergy
    and the HEnergy. 
    """
    
    def __init__(self, energy_conversion_factor, screening, radius=None):
        super().__init__(energy_conversion_factor)
        if screening is None:
            raise ValueError("Screened Coulomb requires specification of a screening type.")
        if radius is None:
            raise ValueError("Screened Coulomb requires specification of a radius")

        if isinstance(screening, type):
            screening = screening()

        self.radius = radius

        self.screening = screening
        self.bond_summer = pairs.MolPairSummer()

    def forward(self, charges, pair_dist, pair_first, pair_second, mol_index, n_molecules):
        screening = self.screening(pair_dist, self.radius).unsqueeze(1)
        screening = torch.where((pair_dist < self.radius).unsqueeze(1), screening, torch.zeros_like(screening))

        # Voltage pairs for per-atom energy
        voltage_pairs = self.energy_conversion_factor * (charges[pair_second] / pair_dist.unsqueeze(1)) 
        voltage_pairs = voltage_pairs * screening 
        n_atoms, _ = charges.shape
        voltage_atom = torch.zeros((n_atoms, 1), device=charges.device, dtype=charges.dtype)
        voltage_atom.index_add_(0, pair_first, voltage_pairs) 
        coulomb_atoms = 0.5 * voltage_atom * charges
        coulomb_molecule = self.summer(coulomb_atoms, mol_index, n_molecules)

        return coulomb_molecule, coulomb_atoms, voltage_atom


class CombineScreenings(torch.nn.Module):
    """ Returns products of different screenings for Screened Coulomb Interactions.
    """
    def __init__(self, screening_list):
        super().__init__()
        self.SL = torch.nn.ModuleList(screening_list)

    def forward(self, pair_dist, radius):
        """ Product of different screenings applied to pair_dist upto radius.

        :param pair_dist: torch.tensor, dtype=float64: 'Neighborlist' distances for coulomb energies.
        :param radius: Maximum radius that Screened-Coulomb is evaluated upto.
        :return screening: Weights for screening for all pair_dist.
        """
        screening = None

        for s in self.SL:
            if screening is None:
                screening = s(pair_dist=pair_dist, radius=radius)
            else:
                screening = screening * s(pair_dist=pair_dist, radius=radius)

        return screening


class AlphaScreening(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha


# Note: This is somewhat incomplete as it does not include a k-space contribution -- more is needed
class EwaldRealSpaceScreening(AlphaScreening):
    def __init__(self, alpha):
        warnings.warn("Ewald implementation incomplete, does not include k-space contributions.")
        super().__init__(alpha)

    def forward(self, pair_dist, radius):
        q = pair_dist / radius
        eta = self.alpha * radius
        return torch.erfc(eta * q)


# Note: typically
class WolfScreening(AlphaScreening):
    def __init__(self, alpha):
        warnings.warn("Wolf implemnetation uses exact derivative of the potential.")
        super().__init__(alpha)

    def forward(self, pair_dist, radius):
        q = pair_dist / radius
        eta = self.alpha * radius
        return torch.erfc(eta * q) - q * torch.erfc(eta)


class LocalDampingCosine(AlphaScreening):
    """ Local damping using complement of the hipnn cutoff function. ('glue-on' method)
    g =     1 if pair_dist > R_cutoff
            1 - [cos(\pi/2 * dist * R_cutoff)]^2  otherwise
    """
    def __init__(self, alpha): 
        """ 
        :param alpha: R_cutoff for glue-on function to ensure 
            smooth crossover from hipnn energy to long-range coulomb energy.  
        """
        super().__init__(alpha) 


    def forward(self, pair_dist, radius):
        """
        :param pair_dist: torch.tensor, dtype=float64: 'Neighborlist' distances for coulomb energies.
        :param radius: Maximum radius that Screened-Coulomb is evaluated upto. 
        :return screening: Weights for screening for each pair.
        """
        pi = torch.tensor([3.141592653589793238], device=pair_dist.device)        
        screening = torch.subtract(torch.tensor([1.0], device=pair_dist.device), torch.square(torch.cos(0.5*pi*pair_dist/self.alpha)))
    
        # pair_dist greater than cut-off; no local-damping. 
        screening = torch.where((pair_dist<self.alpha), screening, torch.ones_like(screening))
        
        return screening


class QScreening(torch.nn.Module):
    def __init__(self, p_value):
        super().__init__()
        self.p_value = p_value

    @property
    def p_value(self):
        return self._p_value

    @p_value.setter
    def p_value(self, value):
        value = int(value)
        self._p_value = value
        powers = torch.arange(1, value + 1, dtype=torch.long).unsqueeze(0)
        self.register_buffer("powers", powers)

    def forward(self, pair_dist, radius):
        q = pair_dist / radius
        q_factors = 1 - torch.pow(q.unsqueeze(1), self.powers)
        product = q_factors.prod(dim=1)
        return product


class PerAtom(torch.nn.Module):
    def forward(self, features, species):
        n_atoms = (species != 0).type(features.dtype).sum(dim=1)
        return features / n_atoms.unsqueeze(1)


class VecMag(torch.nn.Module):
    def forward(self, vector_feature):
        return torch.norm(vector_feature, dim=1)


class CombineEnergy(torch.nn.Module):
    """
    Combines the energies (molecular and atom energies) from two different 
    nodes, e.g. HEnergy, Coulomb, or ScreenedCoulomb Energy Nodes. 
    """
    def __init__(self):
        super().__init__()
        self.summer = indexers.MolSummer()

    def forward(self, atom_energy_1, atom_energy_2, mol_index, n_molecules):
        """
        :param: atom_energy_1 per-atom energy from first node. 
        :param: atom_energy_2 per atom energy from second node. 
        :param: mol_index the molecular index for atoms in the batch
        :param: total number of molecules in the batch
        :return: Total Energy
        """
        total_atom_energy = atom_energy_1 + atom_energy_2
        mol_energy = self.summer(total_atom_energy, mol_index, n_molecules)
        
        return mol_energy, total_atom_energy

class ZBLPotential(torch.nn.Module):
    expo_a = 0.23
    a0 = 0.46850
    coeff = torch.tensor([0.02817, 0.28022, 0.50986, 0.18175])  # for phi(xij/a)
    expo = torch.tensor([0.20162, 0.40290, 0.94229, 3.19980])

    def __init__(self, r_inner, r_outer):
        super().__init__()
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _Phi(self, r, a):
        phi = 0
        for i in range((self.coeff).shape[0]):
            phi += self.coeff[i] * torch.exp(-self.expo[i] * r/a)
        return phi

    def _dPhidr(self, r, a):
        dPhir = 0
        for i in range((self.coeff).shape[0]):
            dPhir += (self.coeff[i] * (-self.expo[i]/a) * torch.exp(-self.expo[i] * r/a))
        return dPhir

    def _d2Phidr2(self, r, a):
        dPhir2 = 0
        for i in range((self.coeff).shape[0]):
            dPhir2 += (self.coeff[i] * ((self.expo[i]/a)**2) * torch.exp(-self.expo[i] * r/a))
        return dPhir2

    def _ZBLE(self, r, a):
        return (1/r) * (self._Phi(r, a))

    def _dZBLEdr(self, r, a):
        Phi = self._Phi(r, a)
        dPhi = self._dPhidr(r, a)
        return 1/r * (-Phi/r + dPhi)

    def _d2ZBLEdr2(self, r, a):
        Phi = self._Phi(r, a)
        dPhi = self._dPhidr(r, a)
        dPhi2 = self._d2Phidr2(r, a)
        return (1/r) * (dPhi2 - 2.0*dPhi*(1./r) + 2.0*Phi*(1./r)**2)

    def forward(self, r, pair_first, pair_second, species):#r, zi, zj):
        #Construct zi and zj
        zi = species[pair_first].squeeze()
        zj = species[pair_second].squeeze()
        #print("VICTORY",r.shape,zi.shape,zj.shape)

        prefixConst = 14.399645478425668 #e*e/(4*pi*epsilon0)  =  14.399645478425668  eV/Ang #1.112 650 055 45 x 10-10 F/m
        zizj = prefixConst * zi * zj
        a = self.a0 / (zi ** self.expo_a + zj ** self.expo_a)

        #Cake the switching function coeffs
        tc = self.r_outer-self.r_inner
        C = -self._ZBLE(self.r_outer, a) + (tc/2.) * self._dZBLEdr(self.r_outer, a) - (1/12.) * (tc ** 2) * self._d2ZBLEdr2(self.r_outer, a)
        B = (2.0 * self._dZBLEdr(self.r_outer, a) - tc * self._d2ZBLEdr2(self.r_outer, a)) / tc ** 3
        A = (-3.0 * self._dZBLEdr(self.r_outer, a) + tc * self._d2ZBLEdr2(self.r_outer, a)) / tc ** 2
        #Consider different conditions
        option_0 = torch.zeros_like(r)
        option_1 = zizj * C
        option_2 = zizj * (self._ZBLE(r, a) + (A/3.) * (r-self.r_inner) ** 3 + (B/4.0) * (r-self.r_inner) ** 4 + C)
        pair_output = torch.where(r>self.r_outer,
                                option_0,
                                torch.where(r<self.r_inner,
                                            option_1,
                                            option_2,
                                            )
                                )
        atom_output = torch.zeros(species.shape[0], device=pair_output.device, dtype=pair_output.dtype)
        atom_output.index_add_(0, pair_first, pair_output)
        return pair_output, atom_output


