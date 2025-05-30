/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2012-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Modeller.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <set>

using namespace OpenMM;
using namespace std;

/**
 * Internal implementation class for Modeller
 */
class OpenMM::ModellerImpl {
public:
    vector<Modeller::Chain> chains;
    vector<Modeller::Residue> residues;
    vector<Modeller::Atom> atoms;
    vector<Modeller::Bond> bonds;
    vector<Vec3> positions;
    vector<Vec3> periodicBoxVectors;
    
    // Fast lookup structures
    unordered_map<int, int> atomIndexMap;        // original index -> current index
    unordered_map<int, int> residueIndexMap;     // original index -> current index
    unordered_map<int, int> chainIndexMap;       // original index -> current index
    
    void rebuildIndices() {
        // Update atom indices and create mapping
        atomIndexMap.clear();
        for (int i = 0; i < (int)atoms.size(); ++i) {
            atomIndexMap[atoms[i].index] = i;
            atoms[i].index = i;
        }
        
        // Update residue indices and create mapping
        residueIndexMap.clear();
        for (int i = 0; i < (int)residues.size(); ++i) {
            residueIndexMap[residues[i].index] = i;
            residues[i].index = i;
            
            // Update atom references to residue indices
            for (int atomIdx : residues[i].atomIndices) {
                if (atomIdx < (int)atoms.size()) {
                    atoms[atomIdx].residueIndex = i;
                }
            }
        }
        
        // Update chain indices and create mapping
        chainIndexMap.clear();
        for (int i = 0; i < (int)chains.size(); ++i) {
            chainIndexMap[chains[i].index] = i;
            chains[i].index = i;
            
            // Update residue references to chain indices
            for (int resIdx : chains[i].residueIndices) {
                if (resIdx < (int)residues.size()) {
                    residues[resIdx].chainIndex = i;
                }
            }
        }
        
        // Update bond atom indices
        for (auto& bond : bonds) {
            auto it1 = atomIndexMap.find(bond.atom1Index);
            auto it2 = atomIndexMap.find(bond.atom2Index);
            if (it1 != atomIndexMap.end() && it2 != atomIndexMap.end()) {
                bond.atom1Index = it1->second;
                bond.atom2Index = it2->second;
            }
        }
    }
    
    void removeDeletedItems(const unordered_set<int>& atomsToDelete,
                           const unordered_set<int>& residuesToDelete,
                           const unordered_set<int>& chainsToDelete,
                           const set<pair<int,int>>& bondsToDelete) {
        // Remove bonds first
        bonds.erase(remove_if(bonds.begin(), bonds.end(),
            [&](const Modeller::Bond& bond) {
                return bondsToDelete.count({bond.atom1Index, bond.atom2Index}) ||
                       bondsToDelete.count({bond.atom2Index, bond.atom1Index}) ||
                       atomsToDelete.count(bond.atom1Index) ||
                       atomsToDelete.count(bond.atom2Index);
            }), bonds.end());
        
        // Remove atoms and their positions
        vector<Vec3> newPositions;
        auto atomIt = atoms.begin();
        auto posIt = positions.begin();
        while (atomIt != atoms.end()) {
            if (atomsToDelete.count(atomIt->index) || 
                residuesToDelete.count(atomIt->residueIndex)) {
                atomIt = atoms.erase(atomIt);
                if (posIt != positions.end()) {
                    posIt = positions.erase(posIt);
                }
            } else {
                // Update residue's atom list
                int residueIdx = atomIt->residueIndex;
                if (residueIdx < (int)residues.size()) {
                    auto& atomIndices = residues[residueIdx].atomIndices;
                    atomIndices.erase(remove(atomIndices.begin(), atomIndices.end(), atomIt->index), atomIndices.end());
                    atomIndices.push_back(atomIt->index); // Add back with updated reference
                }
                
                ++atomIt;
                if (posIt != positions.end()) {
                    ++posIt;
                }
                newPositions.push_back(*(posIt-1));
            }
        }
        positions = newPositions;
        
        // Remove residues
        residues.erase(remove_if(residues.begin(), residues.end(),
            [&](const Modeller::Residue& residue) {
                bool shouldDelete = residuesToDelete.count(residue.index) ||
                                   chainsToDelete.count(residue.chainIndex) ||
                                   residue.atomIndices.empty();
                
                if (shouldDelete && residue.chainIndex < (int)chains.size()) {
                    // Remove from chain's residue list
                    auto& resIndices = chains[residue.chainIndex].residueIndices;
                    resIndices.erase(remove(resIndices.begin(), resIndices.end(), residue.index), resIndices.end());
                }
                return shouldDelete;
            }), residues.end());
        
        // Remove chains
        chains.erase(remove_if(chains.begin(), chains.end(),
            [&](const Modeller::Chain& chain) {
                return chainsToDelete.count(chain.index) || chain.residueIndices.empty();
            }), chains.end());
        
        rebuildIndices();
    }
};

// Static member definitions
map<string, Modeller::ResidueHydrogenData> Modeller::residueHydrogens;
bool Modeller::hasLoadedStandardHydrogens = false;

Modeller::Modeller() : impl(new ModellerImpl()) {
}

Modeller::Modeller(const vector<Chain>& chains,
                   const vector<Residue>& residues,
                   const vector<Atom>& atoms,
                   const vector<Bond>& bonds,
                   const vector<Vec3>& positions,
                   const vector<Vec3>& periodicBoxVectors) : impl(new ModellerImpl()) {
    impl->chains = chains;
    impl->residues = residues;
    impl->atoms = atoms;
    impl->bonds = bonds;
    impl->positions = positions;
    impl->periodicBoxVectors = periodicBoxVectors;
    impl->rebuildIndices();
}

Modeller::~Modeller() {
    delete impl;
}

const vector<Vec3>& Modeller::getPositions() const {
    return impl->positions;
}

void Modeller::setPositions(const vector<Vec3>& positions) {
    if (positions.size() != impl->atoms.size()) {
        throw OpenMMException("Number of positions must match number of atoms");
    }
    impl->positions = positions;
}

void Modeller::getTopology(vector<Chain>& chains,
                          vector<Residue>& residues,
                          vector<Atom>& atoms,
                          vector<Bond>& bonds) const {
    chains = impl->chains;
    residues = impl->residues;
    atoms = impl->atoms;
    bonds = impl->bonds;
}

const vector<Vec3>& Modeller::getPeriodicBoxVectors() const {
    return impl->periodicBoxVectors;
}

void Modeller::setPeriodicBoxVectors(const vector<Vec3>& vectors) {
    impl->periodicBoxVectors = vectors;
}

void Modeller::add(const vector<Chain>& addChains,
                   const vector<Residue>& addResidues,
                   const vector<Atom>& addAtoms,
                   const vector<Bond>& addBonds,
                   const vector<Vec3>& addPositions) {
    if (addAtoms.size() != addPositions.size()) {
        throw OpenMMException("Number of atoms must match number of positions");
    }
    
    // Get current maximum indices
    int maxChainIndex = impl->chains.empty() ? -1 : 
        max_element(impl->chains.begin(), impl->chains.end(),
            [](const Chain& a, const Chain& b) { return a.index < b.index; })->index;
    int maxResidueIndex = impl->residues.empty() ? -1 :
        max_element(impl->residues.begin(), impl->residues.end(),
            [](const Residue& a, const Residue& b) { return a.index < b.index; })->index;
    int maxAtomIndex = impl->atoms.empty() ? -1 :
        max_element(impl->atoms.begin(), impl->atoms.end(),
            [](const Atom& a, const Atom& b) { return a.index < b.index; })->index;
    
    // Index mappings for the new items
    unordered_map<int, int> newChainIndexMap;
    unordered_map<int, int> newResidueIndexMap;
    unordered_map<int, int> newAtomIndexMap;
    
    // Add chains with updated indices
    for (const auto& chain : addChains) {
        Chain newChain = chain;
        newChain.index = ++maxChainIndex;
        newChainIndexMap[chain.index] = newChain.index;
        impl->chains.push_back(newChain);
    }
    
    // Add residues with updated indices and chain references
    for (const auto& residue : addResidues) {
        Residue newResidue = residue;
        newResidue.index = ++maxResidueIndex;
        newResidueIndexMap[residue.index] = newResidue.index;
        
        // Update chain reference
        auto chainIt = newChainIndexMap.find(residue.chainIndex);
        if (chainIt != newChainIndexMap.end()) {
            newResidue.chainIndex = chainIt->second;
            // Add to chain's residue list
            for (auto& chain : impl->chains) {
                if (chain.index == newResidue.chainIndex) {
                    chain.residueIndices.push_back(newResidue.index);
                    break;
                }
            }
        }
        
        impl->residues.push_back(newResidue);
    }
    
    // Add atoms with updated indices and residue references
    for (const auto& atom : addAtoms) {
        Atom newAtom = atom;
        newAtom.index = ++maxAtomIndex;
        newAtomIndexMap[atom.index] = newAtom.index;
        
        // Update residue reference
        auto residueIt = newResidueIndexMap.find(atom.residueIndex);
        if (residueIt != newResidueIndexMap.end()) {
            newAtom.residueIndex = residueIt->second;
            // Add to residue's atom list
            for (auto& residue : impl->residues) {
                if (residue.index == newAtom.residueIndex) {
                    residue.atomIndices.push_back(newAtom.index);
                    break;
                }
            }
        }
        
        impl->atoms.push_back(newAtom);
    }
    
    // Add positions
    impl->positions.insert(impl->positions.end(), addPositions.begin(), addPositions.end());
    
    // Add bonds with updated atom references
    for (const auto& bond : addBonds) {
        Bond newBond = bond;
        auto atom1It = newAtomIndexMap.find(bond.atom1Index);
        auto atom2It = newAtomIndexMap.find(bond.atom2Index);
        
        if (atom1It != newAtomIndexMap.end() && atom2It != newAtomIndexMap.end()) {
            newBond.atom1Index = atom1It->second;
            newBond.atom2Index = atom2It->second;
            impl->bonds.push_back(newBond);
        }
    }
    
    impl->rebuildIndices();
}

void Modeller::deleteItems(const vector<int>& atomsToDelete,
                          const vector<int>& residuesToDelete,
                          const vector<int>& chainsToDelete,
                          const vector<pair<int,int>>& bondsToDelete) {
    
    unordered_set<int> atomsSet(atomsToDelete.begin(), atomsToDelete.end());
    unordered_set<int> residuesSet(residuesToDelete.begin(), residuesToDelete.end());
    unordered_set<int> chainsSet(chainsToDelete.begin(), chainsToDelete.end());
    set<pair<int,int>> bondsSet(bondsToDelete.begin(), bondsToDelete.end());
    
    // Expand deletion sets: if a chain is deleted, delete its residues; if a residue is deleted, delete its atoms
    for (int chainIdx : chainsToDelete) {
        if (chainIdx < (int)impl->chains.size()) {
            for (int resIdx : impl->chains[chainIdx].residueIndices) {
                residuesSet.insert(resIdx);
            }
        }
    }
    
    for (int resIdx : residuesToDelete) {
        if (resIdx < (int)impl->residues.size()) {
            for (int atomIdx : impl->residues[resIdx].atomIndices) {
                atomsSet.insert(atomIdx);
            }
        }
    }
    
    // Also add residues and chains to deletion if all their atoms/residues are being deleted
    for (auto& residue : impl->residues) {
        bool allAtomsDeleted = true;
        for (int atomIdx : residue.atomIndices) {
            if (atomsSet.find(atomIdx) == atomsSet.end()) {
                allAtomsDeleted = false;
                break;
            }
        }
        if (allAtomsDeleted && !residue.atomIndices.empty()) {
            residuesSet.insert(residue.index);
        }
    }
    
    for (auto& chain : impl->chains) {
        bool allResiduesDeleted = true;
        for (int resIdx : chain.residueIndices) {
            if (residuesSet.find(resIdx) == residuesSet.end()) {
                allResiduesDeleted = false;
                break;
            }
        }
        if (allResiduesDeleted && !chain.residueIndices.empty()) {
            chainsSet.insert(chain.index);
        }
    }
    
    impl->removeDeletedItems(atomsSet, residuesSet, chainsSet, bondsSet);
}

void Modeller::deleteWater() {
    vector<int> residuesToDelete;
    
    for (int i = 0; i < (int)impl->residues.size(); ++i) {
        const auto& residue = impl->residues[i];
        if (residue.name == "HOH" || residue.name == "WAT") {
            residuesToDelete.push_back(i);
        }
    }
    
    if (!residuesToDelete.empty()) {
        deleteItems(vector<int>(), residuesToDelete, vector<int>(), vector<pair<int,int>>());
    }
}

int Modeller::getNumAtoms() const {
    return (int)impl->atoms.size();
}

int Modeller::getNumResidues() const {
    return (int)impl->residues.size();
}

int Modeller::getNumChains() const {
    return (int)impl->chains.size();
}

int Modeller::getNumBonds() const {
    return (int)impl->bonds.size();
}

void Modeller::loadHydrogenDefinitions(const map<string, ResidueHydrogenData>& residueHydrogenData) {
    residueHydrogens = residueHydrogenData;
    hasLoadedStandardHydrogens = true;
}

const map<string, Modeller::ResidueHydrogenData>& Modeller::getHydrogenDefinitions() {
    return residueHydrogens;
}

bool Modeller::addHydrogens(vector<string>& selectedVariants,
                           double pH,
                           const vector<string>& variants) {
    // For now, this is a placeholder that delegates to Python for complex logic
    // TODO: Implement full C++ version of hydrogen addition algorithm
    
    // Ensure hydrogen definitions are loaded
    if (residueHydrogens.empty()) {
        return false; // Cannot proceed without hydrogen definitions
    }
    
    // Initialize selectedVariants array
    selectedVariants.clear();
    selectedVariants.resize(impl->residues.size());
    
    // For each residue, determine the appropriate variant based on pH and other factors
    for (size_t i = 0; i < impl->residues.size(); ++i) {
        const auto& residue = impl->residues[i];
        string variant;
        
        // Use provided variant if available
        if (i < variants.size() && !variants[i].empty()) {
            variant = variants[i];
        } else {
            // Auto-select variant based on pH and residue type
            // This is a simplified version - the full algorithm is complex
            auto hydrogenDataIt = residueHydrogens.find(residue.name);
            if (hydrogenDataIt != residueHydrogens.end()) {
                // Simple pH-based selection for common amino acids
                if (residue.name == "ASP") {
                    variant = (pH < 3.9) ? "ASH" : "ASP";
                } else if (residue.name == "GLU") {
                    variant = (pH < 4.3) ? "GLH" : "GLU";
                } else if (residue.name == "HIS") {
                    variant = (pH > 6.0) ? "HIP" : "HID"; // Simplified - real version checks hydrogen bonds
                } else if (residue.name == "LYS") {
                    variant = (pH < 10.5) ? "LYS" : "LYN";
                } else if (residue.name == "CYS") {
                    variant = "CYS"; // TODO: Check for disulfide bonds
                } else {
                    // Use first available variant
                    if (!hydrogenDataIt->second.variants.empty()) {
                        variant = hydrogenDataIt->second.variants[0];
                    }
                }
            }
        }
        
        selectedVariants[i] = variant;
    }
    
    // TODO: Actually add the hydrogen atoms and positions
    // For now, return false to indicate this is not fully implemented
    return false;
}