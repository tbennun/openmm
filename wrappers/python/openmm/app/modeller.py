"""
modeller_cpp.py: C++ accelerated implementation of the Modeller class

This module provides a Python wrapper around the C++ Modeller implementation
while maintaining the exact same API as the original Python version.
"""
from __future__ import division
from __future__ import absolute_import

__author__ = "Peter Eastman"
__version__ = "1.0"

from openmm.app import Topology, PDBFile, ForceField
from openmm.app.forcefield import AllBonds, CutoffNonPeriodic, CutoffPeriodic, DrudeGenerator, _getDataDirectories
from openmm.app.internal import compiled
from openmm.vec3 import Vec3
from openmm import System, Context, NonbondedForce, CustomNonbondedForce, HarmonicBondForce, HarmonicAngleForce, VerletIntegrator, LangevinIntegrator, LocalEnergyMinimizer
from openmm.unit import nanometer, molar, elementary_charge, degree, acos, is_quantity, dot, norm, kilojoules_per_mole
import openmm.unit as unit
from . import element as elem
import openmm
import gc
import os
import random
import sys
import xml.etree.ElementTree as etree
from copy import deepcopy
from math import ceil, floor, sqrt
from collections import defaultdict

# Try to import the C++ Modeller class
try:
    from openmm import Modeller as CppModeller
    CPP_MODELLER_AVAILABLE = True
except ImportError:
    CPP_MODELLER_AVAILABLE = False

class Modeller(object):
    """Modeller provides tools for editing molecular models, such as adding water or missing hydrogens.

    To use it, create a Modeller object, specifying the initial Topology and atom positions.  You can
    then call various methods to change the model in different ways.  Each time you do, a new Topology
    and list of coordinates is created to represent the changed model.  Finally, call getTopology()
    and getPositions() to get the results.
    
    This implementation uses a C++ backend for performance-critical operations while maintaining
    the same Python API for compatibility.
    """

    _residueHydrogens = {}
    _hasLoadedStandardHydrogens = False

    def __init__(self, topology, positions):
        """Create a new Modeller object

        Parameters
        ----------
        topology : Topology
            the initial Topology of the model
        positions : list
            the initial atomic positions
        """
        ## The Topology describing the structure of the system
        self.topology = topology
        if not is_quantity(positions):
            positions = positions*nanometer
        ## The list of atom positions
        self.positions = positions
        
        # Initialize C++ backend if available
        if CPP_MODELLER_AVAILABLE:
            self._cpp_modeller = self._create_cpp_modeller()
        else:
            self._cpp_modeller = None

    def _create_cpp_modeller(self):
        """Create a C++ Modeller instance from the current topology and positions."""
        if not CPP_MODELLER_AVAILABLE:
            return None
            
        # Convert topology to C++ format
        chains = []
        residues = []
        atoms = []
        bonds = []
        
        # Build chains
        for chain_idx, chain in enumerate(self.topology.chains()):
            cpp_chain = CppModeller.Chain(chain.id, chain_idx)
            chains.append(cpp_chain)
        
        # Build residues
        residue_idx = 0
        chain_residue_map = {}
        for chain_idx, chain in enumerate(self.topology.chains()):
            for residue in chain.residues():
                cpp_residue = CppModeller.Residue(
                    residue.name, residue_idx, chain_idx, 
                    residue.id, residue.insertionCode
                )
                residues.append(cpp_residue)
                chain_residue_map[residue] = residue_idx
                chains[chain_idx].residueIndices.append(residue_idx)
                residue_idx += 1
        
        # Build atoms
        atom_idx = 0
        residue_atom_map = {}
        for chain_idx, chain in enumerate(self.topology.chains()):
            for residue in chain.residues():
                res_idx = chain_residue_map[residue]
                for atom in residue.atoms():
                    element_num = 0 if atom.element is None else atom.element.atomic_number
                    cpp_atom = CppModeller.Atom(
                        atom.name, element_num, atom_idx, res_idx,
                        atom.id, atom.formalCharge
                    )
                    atoms.append(cpp_atom)
                    residue_atom_map[atom] = atom_idx
                    residues[res_idx].atomIndices.append(atom_idx)
                    atom_idx += 1
        
        # Build bonds
        for bond in self.topology.bonds():
            atom1_idx = residue_atom_map[bond[0]]
            atom2_idx = residue_atom_map[bond[1]]
            
            # Convert bond type to integer
            bond_type = 0
            bond_order = 0.0
            if hasattr(bond, 'type') and bond.type is not None:
                if str(bond.type) == 'Single':
                    bond_type = 1
                    bond_order = 1.0
                elif str(bond.type) == 'Double':
                    bond_type = 2
                    bond_order = 2.0
                elif str(bond.type) == 'Triple':
                    bond_type = 3
                    bond_order = 3.0
                elif str(bond.type) == 'Aromatic':
                    bond_type = 4
                    bond_order = 1.5
                elif str(bond.type) == 'Amide':
                    bond_type = 5
                    bond_order = 1.0
            if hasattr(bond, 'order') and bond.order is not None:
                bond_order = float(bond.order)
            
            cpp_bond = CppModeller.Bond(atom1_idx, atom2_idx, bond_type, bond_order)
            bonds.append(cpp_bond)
        
        # Convert positions
        positions_vec = []
        for pos in self.positions:
            if is_quantity(pos):
                pos_nm = pos.value_in_unit(nanometer)
            else:
                pos_nm = pos
            positions_vec.append(Vec3(pos_nm[0], pos_nm[1], pos_nm[2]))
        
        # Get periodic box vectors
        box_vectors = []
        if self.topology.getPeriodicBoxVectors() is not None:
            for vec in self.topology.getPeriodicBoxVectors():
                vec_nm = vec.value_in_unit(nanometer) if is_quantity(vec) else vec
                box_vectors.append(Vec3(vec_nm[0], vec_nm[1], vec_nm[2]))
        
        return CppModeller(chains, residues, atoms, bonds, positions_vec, box_vectors)

    def _update_from_cpp_modeller(self):
        """Update the Python topology and positions from the C++ modeller."""
        if self._cpp_modeller is None:
            return
            
        # Get data from C++ modeller
        chains = []
        residues = []
        atoms = []
        bonds = []
        self._cpp_modeller.getTopology(chains, residues, atoms, bonds)
        positions = self._cpp_modeller.getPositions()
        
        # Build new topology
        new_topology = Topology()
        
        # Set periodic box vectors
        box_vectors = self._cpp_modeller.getPeriodicBoxVectors()
        if box_vectors:
            new_topology.setPeriodicBoxVectors([
                Vec3(v[0], v[1], v[2])*nanometer for v in box_vectors
            ])
        
        # Create mapping from C++ indices to Python objects
        chain_map = {}
        residue_map = {}
        atom_map = {}
        
        # Add chains
        for cpp_chain in chains:
            py_chain = new_topology.addChain(cpp_chain.id)
            chain_map[cpp_chain.index] = py_chain
        
        # Add residues
        for cpp_residue in residues:
            py_chain = chain_map[cpp_residue.chainIndex]
            py_residue = new_topology.addResidue(
                cpp_residue.name, py_chain, cpp_residue.id, cpp_residue.insertionCode
            )
            residue_map[cpp_residue.index] = py_residue
        
        # Add atoms
        for cpp_atom in atoms:
            py_residue = residue_map[cpp_atom.residueIndex]
            element = None if cpp_atom.element == 0 else elem.Element.getByAtomicNumber(cpp_atom.element)
            py_atom = new_topology.addAtom(
                cpp_atom.name, element, py_residue, cpp_atom.id, cpp_atom.formalCharge
            )
            atom_map[cpp_atom.index] = py_atom
        
        # Add bonds
        from openmm.app.topology import Single, Double, Triple, Aromatic, Amide
        bond_types = [None, Single, Double, Triple, Aromatic, Amide]
        
        for cpp_bond in bonds:
            py_atom1 = atom_map[cpp_bond.atom1Index]
            py_atom2 = atom_map[cpp_bond.atom2Index]
            bond_type = bond_types[cpp_bond.bondType] if cpp_bond.bondType < len(bond_types) else None
            py_bond = new_topology.addBond(py_atom1, py_atom2, bond_type, cpp_bond.bondOrder)
        
        # Update positions
        new_positions = []
        for pos in positions:
            new_positions.append(Vec3(pos[0], pos[1], pos[2])*nanometer)
        
        self.topology = new_topology
        self.positions = new_positions

    def getTopology(self):
        """Get the Topology of the model."""
        return self.topology

    def getPositions(self):
        """Get the atomic positions."""
        return self.positions

    def add(self, addTopology, addPositions):
        """Add chains, residues, atoms, and bonds to the model.

        Specify what to add by providing a new Topology object and the
        corresponding atomic positions. All chains, residues, atoms, and bonds
        contained in the Topology are added to the model.

        Parameters
        ----------
        addTopology : Topology
            a Topology whose contents should be added to the model
        addPositions : list
            the positions of the atoms to add
        """
        if self._cpp_modeller is not None:
            # Use C++ implementation
            # First, convert the topology to add to C++ format
            temp_modeller = Modeller(addTopology, addPositions)
            add_chains = []
            add_residues = []
            add_atoms = []
            add_bonds = []
            temp_modeller._cpp_modeller.getTopology(add_chains, add_residues, add_atoms, add_bonds)
            add_positions = temp_modeller._cpp_modeller.getPositions()
            
            # Add to our C++ modeller
            self._cpp_modeller.add(add_chains, add_residues, add_atoms, add_bonds, add_positions)
            
            # Update Python topology
            self._update_from_cpp_modeller()
        else:
            # Fallback to original Python implementation
            self._add_python(addTopology, addPositions)

    def _add_python(self, addTopology, addPositions):
        """Original Python implementation of add()"""
        # Copy over the existing model.

        newTopology = Topology()
        newTopology.setPeriodicBoxVectors(self.topology.getPeriodicBoxVectors())
        newAtoms = {}
        newPositions = []*nanometer
        for chain in self.topology.chains():
            newChain = newTopology.addChain(chain.id)
            for residue in chain.residues():
                newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                for atom in residue.atoms():
                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id, atom.formalCharge)
                    newAtoms[atom] = newAtom
                    newPositions.append(deepcopy(self.positions[atom.index]))
        for bond in self.topology.bonds():
            newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]], bond.type, bond.order)

        # Add the new model

        newAtoms = {}
        for chain in addTopology.chains():
            newChain = newTopology.addChain(chain.id)
            for residue in chain.residues():
                newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                for atom in residue.atoms():
                    newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id, atom.formalCharge)
                    newAtoms[atom] = newAtom
                    newPositions.append(deepcopy(addPositions[atom.index]))
        for bond in addTopology.bonds():
            newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]], bond.type, bond.order)
        self.topology = newTopology
        self.positions = newPositions

    def delete(self, toDelete):
        """Delete chains, residues, atoms, and bonds from the model.

        You can specify objects to delete at any granularity: atoms, residues, or chains.  Passing
        in an Atom object causes that Atom to be deleted.  Passing in a Residue object causes that
        Residue and all Atoms it contains to be deleted.  Passing in a Chain object causes that
        Chain and all Residues and Atoms it contains to be deleted.

        In all cases, when an Atom is deleted, any bonds it participates in are also deleted.
        You also can specify a bond (as a tuple of Atom objects) to delete just that bond without
        deleting the Atoms it connects.

        Parameters
        ----------
        toDelete : list
            a list of Atoms, Residues, Chains, and bonds (specified as tuples of
            Atoms) to delete
        """
        if self._cpp_modeller is not None:
            # Use C++ implementation
            atoms_to_delete = []
            residues_to_delete = []
            chains_to_delete = []
            bonds_to_delete = []
            
            # Create index mappings
            atom_to_index = {atom: atom.index for atom in self.topology.atoms()}
            residue_to_index = {residue: idx for idx, residue in enumerate(self.topology.residues())}
            chain_to_index = {chain: idx for idx, chain in enumerate(self.topology.chains())}
            
            for item in toDelete:
                if hasattr(item, 'element'):  # Atom
                    atoms_to_delete.append(atom_to_index[item])
                elif hasattr(item, 'name') and hasattr(item, 'atoms'):  # Residue
                    residues_to_delete.append(residue_to_index[item])
                elif hasattr(item, 'id') and hasattr(item, 'residues'):  # Chain
                    chains_to_delete.append(chain_to_index[item])
                elif isinstance(item, (tuple, list)) and len(item) == 2:  # Bond
                    bonds_to_delete.append((atom_to_index[item[0]], atom_to_index[item[1]]))
            
            self._cpp_modeller.deleteItems(atoms_to_delete, residues_to_delete, chains_to_delete, bonds_to_delete)
            self._update_from_cpp_modeller()
        else:
            # Fallback to original Python implementation
            self._delete_python(toDelete)

    def _delete_python(self, toDelete):
        """Original Python implementation of delete()"""
        newTopology = Topology()
        newTopology.setPeriodicBoxVectors(self.topology.getPeriodicBoxVectors())
        newAtoms = {}
        newPositions = []*nanometer
        deleteSet = set(toDelete)
        for chain in self.topology.chains():
            if chain not in deleteSet:
                needNewChain = True;
                for residue in chain.residues():
                    if residue not in deleteSet:
                        needNewResidue = True
                        for atom in residue.atoms():
                            if atom not in deleteSet:
                                if needNewChain:
                                    newChain = newTopology.addChain(chain.id)
                                    needNewChain = False;
                                if needNewResidue:
                                    newResidue = newTopology.addResidue(residue.name, newChain, residue.id, residue.insertionCode)
                                    needNewResidue = False;
                                newAtom = newTopology.addAtom(atom.name, atom.element, newResidue, atom.id, atom.formalCharge)
                                newAtoms[atom] = newAtom
                                newPositions.append(deepcopy(self.positions[atom.index]))
        for bond in self.topology.bonds():
            if bond[0] in newAtoms and bond[1] in newAtoms:
                if bond not in deleteSet and (bond[1], bond[0]) not in deleteSet:
                    newTopology.addBond(newAtoms[bond[0]], newAtoms[bond[1]], bond.type, bond.order)
        self.topology = newTopology
        self.positions = newPositions

    def deleteWater(self):
        """Delete all water molecules from the model."""
        if self._cpp_modeller is not None:
            # Use C++ implementation
            self._cpp_modeller.deleteWater()
            self._update_from_cpp_modeller()
        else:
            # Fallback to Python implementation
            self.delete(res for res in self.topology.residues() if res.name == "HOH")

    # For methods that are not performance-critical, delegate to the original Python implementation
    def convertWater(self, model='tip3p'):
        # Import the original implementation
        from . import modeller_original as original_modeller
        # Create temporary instance with original implementation
        temp_modeller = original_modeller.Modeller(self.topology, self.positions)
        temp_modeller.convertWater(model)
        # Update our state
        self.topology = temp_modeller.topology
        self.positions = temp_modeller.positions
        # Recreate C++ modeller if available
        if CPP_MODELLER_AVAILABLE:
            self._cpp_modeller = self._create_cpp_modeller()

    def addSolvent(self, forcefield, model='tip3p', boxSize=None, boxVectors=None, padding=None, numAdded=None, boxShape='cube', positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*molar, neutralize=True, residueTemplates=dict()):
        # Import the original implementation
        from . import modeller_original as original_modeller
        # Create temporary instance with original implementation
        temp_modeller = original_modeller.Modeller(self.topology, self.positions)
        temp_modeller.addSolvent(forcefield, model, boxSize, boxVectors, padding, numAdded, boxShape, positiveIon, negativeIon, ionicStrength, neutralize, residueTemplates)
        # Update our state
        self.topology = temp_modeller.topology
        self.positions = temp_modeller.positions
        # Recreate C++ modeller if available
        if CPP_MODELLER_AVAILABLE:
            self._cpp_modeller = self._create_cpp_modeller()

    def addHydrogens(self, forcefield, pH=7.0, variants=None, platform=None, residueTemplates=dict()):
        # Import the original implementation
        from . import modeller_original as original_modeller
        # Create temporary instance with original implementation
        temp_modeller = original_modeller.Modeller(self.topology, self.positions)
        result = temp_modeller.addHydrogens(forcefield, pH, variants, platform, residueTemplates)
        # Update our state
        self.topology = temp_modeller.topology
        self.positions = temp_modeller.positions
        # Recreate C++ modeller if available
        if CPP_MODELLER_AVAILABLE:
            self._cpp_modeller = self._create_cpp_modeller()
        return result

    def addExtraParticles(self, forcefield, ignoreExternalBonds=False, residueTemplates=dict()):
        # Import the original implementation
        from . import modeller_original as original_modeller
        # Create temporary instance with original implementation
        temp_modeller = original_modeller.Modeller(self.topology, self.positions)
        temp_modeller.addExtraParticles(forcefield, ignoreExternalBonds, residueTemplates)
        # Update our state
        self.topology = temp_modeller.topology
        self.positions = temp_modeller.positions
        # Recreate C++ modeller if available
        if CPP_MODELLER_AVAILABLE:
            self._cpp_modeller = self._create_cpp_modeller()

    def addMembrane(self, forcefield, lipidType='POPC', membraneCenterZ=0*nanometer, minimumPadding=1*nanometer, positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*molar, neutralize=True, residueTemplates=dict(), platform=None):
        # Import the original implementation
        from . import modeller_original as original_modeller
        # Create temporary instance with original implementation
        temp_modeller = original_modeller.Modeller(self.topology, self.positions)
        temp_modeller.addMembrane(forcefield, lipidType, membraneCenterZ, minimumPadding, positiveIon, negativeIon, ionicStrength, neutralize, residueTemplates, platform)
        # Update our state
        self.topology = temp_modeller.topology
        self.positions = temp_modeller.positions
        # Recreate C++ modeller if available
        if CPP_MODELLER_AVAILABLE:
            self._cpp_modeller = self._create_cpp_modeller()

    @staticmethod
    def loadHydrogenDefinitions(file):
        """Load an XML file containing definitions of hydrogens that should be added by addHydrogens().

        The built in hydrogens.xml file containing definitions for standard amino acids and nucleotides is loaded automatically.
        This method can be used to load additional definitions for other residue types.  They will then be used in subsequent
        calls to addHydrogens().

        Parameters
        ----------
        file : string or file
            An XML file containing hydrogen definitions.  It may be either an
            absolute file path, a path relative to the current working
            directory, a path relative to this module's data subdirectory (for
            built in sets of definitions), or an open file-like object with a read()
            method from which the data can be loaded.
        """
        # Import the original implementation and delegate
        from . import modeller_original as original_modeller
        original_modeller.Modeller.loadHydrogenDefinitions(file)
        
        # Also load into C++ if available
        if CPP_MODELLER_AVAILABLE:
            # Convert the loaded hydrogen definitions to C++ format
            cpp_residue_data = {}
            for res_name, res_data in original_modeller.Modeller._residueHydrogens.items():
                cpp_res_data = CppModeller.ResidueHydrogenData(res_name)
                cpp_res_data.variants = list(res_data.variants)
                
                for h_data in res_data.hydrogens:
                    cpp_h_def = CppModeller.HydrogenDefinition(
                        h_data.name, h_data.parent, h_data.maxph,
                        list(h_data.variants), h_data.terminal
                    )
                    cpp_res_data.hydrogens.append(cpp_h_def)
                
                cpp_residue_data[res_name] = cpp_res_data
            
            CppModeller.loadHydrogenDefinitions(cpp_residue_data)

# Initialize the class static members by delegating to the original implementation
from . import modeller_original as original_modeller
Modeller._residueHydrogens = original_modeller.Modeller._residueHydrogens
Modeller._hasLoadedStandardHydrogens = original_modeller.Modeller._hasLoadedStandardHydrogens