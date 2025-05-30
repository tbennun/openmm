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
#include <cmath>

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
    // Ensure hydrogen definitions are loaded
    if (residueHydrogens.empty()) {
        return false; // Cannot proceed without hydrogen definitions
    }
    
    // Initialize selectedVariants array
    selectedVariants.clear();
    selectedVariants.resize(impl->residues.size());
    
    // Build bonding information
    unordered_map<int, vector<int>> bondedAtoms;
    for (const auto& atom : impl->atoms) {
        bondedAtoms[atom.index] = vector<int>();
    }
    for (const auto& bond : impl->bonds) {
        bondedAtoms[bond.atom1Index].push_back(bond.atom2Index);
        bondedAtoms[bond.atom2Index].push_back(bond.atom1Index);
    }
    
    // Track new atoms to add
    vector<Atom> newAtoms;
    vector<Vec3> newPositions;
    vector<Bond> newBonds;
    
    // For each residue, determine variant and add hydrogens
    for (size_t resIdx = 0; resIdx < impl->residues.size(); ++resIdx) {
        const auto& residue = impl->residues[resIdx];
        string variant;
        
        // Use provided variant if available
        if (resIdx < variants.size() && !variants[resIdx].empty()) {
            variant = variants[resIdx];
        } else {
            // Auto-select variant based on pH and residue type
            auto hydrogenDataIt = residueHydrogens.find(residue.name);
            if (hydrogenDataIt != residueHydrogens.end()) {
                variant = selectVariantForResidue(residue, pH, bondedAtoms);
            } else {
                variant = residue.name; // Use residue name as default
            }
        }
        
        selectedVariants[resIdx] = variant;
        
        // Add hydrogens for this variant
        addHydrogensForVariant(residue, variant, bondedAtoms, newAtoms, newPositions, newBonds);
    }
    
    // Add all new atoms, positions, and bonds
    if (!newAtoms.empty()) {
        int baseAtomIndex = impl->atoms.size();
        
        // Update atom indices and residue references
        for (size_t i = 0; i < newAtoms.size(); ++i) {
            newAtoms[i].index = baseAtomIndex + i;
            
            // Add to residue's atom list
            int resIdx = newAtoms[i].residueIndex;
            if (resIdx < (int)impl->residues.size()) {
                impl->residues[resIdx].atomIndices.push_back(newAtoms[i].index);
            }
        }
        
        // Update bond indices
        for (auto& bond : newBonds) {
            if (bond.atom1Index >= baseAtomIndex) bond.atom1Index += baseAtomIndex;
            if (bond.atom2Index >= baseAtomIndex) bond.atom2Index += baseAtomIndex;
        }
        
        // Add to main vectors
        impl->atoms.insert(impl->atoms.end(), newAtoms.begin(), newAtoms.end());
        impl->positions.insert(impl->positions.end(), newPositions.begin(), newPositions.end());
        impl->bonds.insert(impl->bonds.end(), newBonds.begin(), newBonds.end());
        
        return true;
    }
    
    return false; // No hydrogens were added
}

string Modeller::selectVariantForResidue(const Residue& residue, double pH,
                                        const unordered_map<int, vector<int>>& bondedAtoms) {
    // Handle special cases for variant selection
    if (residue.name == "CYS") {
        // Check for disulfide bonds by looking for sulfur atoms bonded to other residues
        for (int atomIdx : residue.atomIndices) {
            if (atomIdx < (int)impl->atoms.size()) {
                const auto& atom = impl->atoms[atomIdx];
                if (atom.element == 16) { // Sulfur
                    for (int bondedIdx : bondedAtoms.at(atomIdx)) {
                        if (bondedIdx < (int)impl->atoms.size()) {
                            const auto& bondedAtom = impl->atoms[bondedIdx];
                            if (bondedAtom.residueIndex != residue.index) {
                                return "CYX"; // Disulfide bond found
                            }
                        }
                    }
                }
            }
        }
        return "CYS";
    }
    
    // pH-based selection for ionizable residues
    if (residue.name == "ASP") {
        return (pH < 3.9) ? "ASH" : "ASP";
    } else if (residue.name == "GLU") {
        return (pH < 4.3) ? "GLH" : "GLU";
    } else if (residue.name == "LYS") {
        return (pH < 10.5) ? "LYS" : "LYN";
    } else if (residue.name == "HIS") {
        return selectHistidineVariant(residue, pH, bondedAtoms);
    }
    
    // Default: use residue name or first variant
    auto hydrogenDataIt = residueHydrogens.find(residue.name);
    if (hydrogenDataIt != residueHydrogens.end() && !hydrogenDataIt->second.variants.empty()) {
        return hydrogenDataIt->second.variants[0];
    }
    
    return residue.name;
}

string Modeller::selectHistidineVariant(const Residue& residue, double pH,
                                       const unordered_map<int, vector<int>>& bondedAtoms) {
    if (pH <= 6.5) {
        return "HID"; // Neutral at low pH
    }
    
    // Check for existing hydrogens on ND1 and NE2
    bool nd1HasHydrogen = false, ne2HasHydrogen = false;
    
    for (int atomIdx : residue.atomIndices) {
        if (atomIdx < (int)impl->atoms.size()) {
            const auto& atom = impl->atoms[atomIdx];
            if (atom.name == "ND1" || atom.name == "NE2") {
                for (int bondedIdx : bondedAtoms.at(atomIdx)) {
                    if (bondedIdx < (int)impl->atoms.size()) {
                        const auto& bondedAtom = impl->atoms[bondedIdx];
                        if (bondedAtom.element == 1) { // Hydrogen
                            if (atom.name == "ND1") nd1HasHydrogen = true;
                            if (atom.name == "NE2") ne2HasHydrogen = true;
                        }
                    }
                }
            }
        }
    }
    
    if (nd1HasHydrogen && ne2HasHydrogen) return "HIP";
    if (nd1HasHydrogen) return "HID";
    if (ne2HasHydrogen) return "HIE";
    
    // Default to HID for simplicity (real implementation would check hydrogen bonding)
    return "HID";
}

void Modeller::addHydrogensForVariant(const Residue& residue, const string& variant,
                                     const unordered_map<int, vector<int>>& bondedAtoms,
                                     vector<Atom>& newAtoms, vector<Vec3>& newPositions,
                                     vector<Bond>& newBonds) {
    auto hydrogenDataIt = residueHydrogens.find(variant);
    if (hydrogenDataIt == residueHydrogens.end()) {
        hydrogenDataIt = residueHydrogens.find(residue.name);
        if (hydrogenDataIt == residueHydrogens.end()) {
            return; // No hydrogen definitions for this residue/variant
        }
    }
    
    const auto& hydrogenData = hydrogenDataIt->second;
    
    // Create atom name to index mapping for this residue
    unordered_map<string, int> atomNameToIndex;
    for (int atomIdx : residue.atomIndices) {
        if (atomIdx < (int)impl->atoms.size()) {
            const auto& atom = impl->atoms[atomIdx];
            atomNameToIndex[atom.name] = atomIdx;
        }
    }
    
    // Add each hydrogen defined for this variant
    for (const auto& hydrogenDef : hydrogenData.hydrogens) {
        // Skip if this hydrogen is pH-dependent and conditions aren't met
        if (hydrogenDef.maxph > 0 && hydrogenDef.maxph < 14.0) {
            // For now, skip pH-dependent hydrogens - this can be enhanced later
            continue;
        }
        
        // Check variant specificity 
        if (!hydrogenDef.variants.empty()) {
            bool variantMatches = false;
            for (const string& variantName : hydrogenDef.variants) {
                if (variantName == variant) {
                    variantMatches = true;
                    break;
                }
            }
            if (!variantMatches) {
                continue; // This hydrogen is not for this variant
            }
        }
        
        // Find parent atom
        auto parentIt = atomNameToIndex.find(hydrogenDef.parent);
        if (parentIt == atomNameToIndex.end()) {
            continue; // Parent atom not found
        }
        
        int parentAtomIdx = parentIt->second;
        
        // Check if hydrogen already exists
        bool hydrogenExists = false;
        auto bondedIt = bondedAtoms.find(parentAtomIdx);
        if (bondedIt != bondedAtoms.end()) {
            for (int bondedIdx : bondedIt->second) {
                if (bondedIdx < (int)impl->atoms.size()) {
                    const auto& bondedAtom = impl->atoms[bondedIdx];
                    if (bondedAtom.element == 1 && bondedAtom.name == hydrogenDef.name) {
                        hydrogenExists = true;
                        break;
                    }
                }
            }
        }
        
        if (hydrogenExists) {
            continue; // Hydrogen already present
        }
        
        // Calculate hydrogen position using improved geometry
        Vec3 hydrogenPos = calculateHydrogenPosition(parentAtomIdx, hydrogenDef.name, bondedAtoms);
        
        // Create new hydrogen atom
        int newAtomIndex = impl->atoms.size() + newAtoms.size();
        Atom newHydrogen(hydrogenDef.name, 1, newAtomIndex, residue.index);
        
        newAtoms.push_back(newHydrogen);
        newPositions.push_back(hydrogenPos);
        
        // Create bond to parent
        Bond newBond(parentAtomIdx, newAtomIndex, 1, 1.0); // Single bond
        newBonds.push_back(newBond);
    }
}

Vec3 Modeller::calculateHydrogenPosition(int parentAtomIdx,
                                        const unordered_map<int, vector<int>>& bondedAtoms) {
    return calculateHydrogenPosition(parentAtomIdx, "", bondedAtoms);
}

Vec3 Modeller::calculateHydrogenPosition(int parentAtomIdx, const string& hydrogenName,
                                        const unordered_map<int, vector<int>>& bondedAtoms) {
    if (parentAtomIdx >= (int)impl->atoms.size()) {
        return Vec3(0, 0, 0);
    }
    
    const Vec3& parentPos = impl->positions[parentAtomIdx];
    const auto& parentAtom = impl->atoms[parentAtomIdx];
    
    // Element-specific bond lengths (in nanometers)
    double bondLength = 0.1; // Default ~1 Angstrom
    switch (parentAtom.element) {
        case 6:  bondLength = 0.109; break; // C-H ~1.09 Å
        case 7:  bondLength = 0.101; break; // N-H ~1.01 Å  
        case 8:  bondLength = 0.096; break; // O-H ~0.96 Å
        case 16: bondLength = 0.134; break; // S-H ~1.34 Å
        default: bondLength = 0.1; break;
    }
    
    auto bondedIt = bondedAtoms.find(parentAtomIdx);
    if (bondedIt == bondedAtoms.end()) {
        // No bonded atoms, place hydrogen arbitrarily
        return parentPos + Vec3(bondLength, 0, 0);
    }
    
    const vector<int>& bondedIndices = bondedIt->second;
    
    if (bondedIndices.empty()) {
        // No bonded atoms, place hydrogen arbitrarily
        return parentPos + Vec3(bondLength, 0, 0);
    }
    
    // Get positions of bonded atoms
    vector<Vec3> bondedPositions;
    for (int bondedIdx : bondedIndices) {
        if (bondedIdx < (int)impl->positions.size()) {
            bondedPositions.push_back(impl->positions[bondedIdx]);
        }
    }
    
    if (bondedPositions.empty()) {
        return parentPos + Vec3(bondLength, 0, 0);
    }
    
    // Calculate placement based on hybridization and geometry
    int numBonded = bondedPositions.size();
    
    if (numBonded == 1) {
        // Linear or sp3 with one bond - place opposite to bonded atom
        Vec3 direction = parentPos - bondedPositions[0];
        double length = sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]);
        if (length > 0) {
            direction = direction * (bondLength/length);
            return parentPos + direction;
        }
        return parentPos + Vec3(bondLength, 0, 0);
    }
    
    if (numBonded == 2) {
        // Trigonal planar or tetrahedral - place in plane or out of plane
        Vec3 v1 = bondedPositions[0] - parentPos;
        Vec3 v2 = bondedPositions[1] - parentPos;
        
        // Normalize vectors
        double len1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
        double len2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);
        if (len1 > 0) v1 = v1 * (1.0/len1);
        if (len2 > 0) v2 = v2 * (1.0/len2);
        
        // Calculate angle bisector (opposite direction)
        Vec3 bisector = -(v1 + v2);
        double bisectorLen = sqrt(bisector[0]*bisector[0] + bisector[1]*bisector[1] + bisector[2]*bisector[2]);
        if (bisectorLen > 0) {
            bisector = bisector * (bondLength/bisectorLen);
            return parentPos + bisector;
        }
        
        // Fallback: place perpendicular to bond plane
        Vec3 cross(v1[1]*v2[2] - v1[2]*v2[1], 
                  v1[2]*v2[0] - v1[0]*v2[2], 
                  v1[0]*v2[1] - v1[1]*v2[0]);
        double crossLen = sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
        if (crossLen > 0) {
            cross = cross * (bondLength/crossLen);
            return parentPos + cross;
        }
    }
    
    if (numBonded >= 3) {
        // Tetrahedral - calculate the missing tetrahedral direction
        Vec3 avgDirection(0, 0, 0);
        for (const Vec3& bondedPos : bondedPositions) {
            Vec3 direction = parentPos - bondedPos;
            double length = sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]);
            if (length > 0) {
                direction = direction * (1.0/length);
                avgDirection = avgDirection + direction;
            }
        }
        
        // Normalize average direction
        double avgLength = sqrt(avgDirection[0]*avgDirection[0] + avgDirection[1]*avgDirection[1] + avgDirection[2]*avgDirection[2]);
        if (avgLength > 0) {
            avgDirection = avgDirection * (bondLength/avgLength);
            return parentPos + avgDirection;
        }
    }
    
    // Default fallback
    return parentPos + Vec3(bondLength, 0, 0);
}

bool Modeller::addSolvent(const string& model,
                         const Vec3& boxSize,
                         const vector<Vec3>& boxVectors,
                         double padding,
                         int numAdded,
                         const string& boxShape,
                         const string& positiveIon,
                         const string& negativeIon,
                         double ionicStrength,
                         bool neutralize) {
    // Determine box vectors - this is the essential part for basic functionality
    vector<Vec3> finalBoxVectors = boxVectors;
    
    if (finalBoxVectors.empty()) {
        if (boxSize[0] > 0 && boxSize[1] > 0 && boxSize[2] > 0) {
            // Use provided box size to create cubic/rectangular box
            if (boxShape == "cube" || boxShape == "rectangular") {
                finalBoxVectors.push_back(Vec3(boxSize[0], 0, 0));
                finalBoxVectors.push_back(Vec3(0, boxSize[1], 0));
                finalBoxVectors.push_back(Vec3(0, 0, boxSize[2]));
            } else {
                // For other shapes, delegate to Python for now
                return false;
            }
        } else if (padding > 0) {
            // Calculate box size from padding
            if (impl->positions.empty()) {
                // No atoms, create minimal box
                double boxDim = max(2*padding, 2.0); // At least 2 nm
                finalBoxVectors.push_back(Vec3(boxDim, 0, 0));
                finalBoxVectors.push_back(Vec3(0, boxDim, 0));
                finalBoxVectors.push_back(Vec3(0, 0, boxDim));
            } else {
                // Calculate bounding box of existing atoms
                Vec3 minPos(1e10, 1e10, 1e10);
                Vec3 maxPos(-1e10, -1e10, -1e10);
                
                for (const auto& pos : impl->positions) {
                    minPos = Vec3(min(minPos[0], pos[0]), min(minPos[1], pos[1]), min(minPos[2], pos[2]));
                    maxPos = Vec3(max(maxPos[0], pos[0]), max(maxPos[1], pos[1]), max(maxPos[2], pos[2]));
                }
                
                // Add padding to all dimensions
                Vec3 size = maxPos - minPos + Vec3(2*padding, 2*padding, 2*padding);
                
                // Ensure minimum size
                double minDim = 2*padding;
                size = Vec3(max(size[0], minDim), max(size[1], minDim), max(size[2], minDim));
                
                if (boxShape == "cube") {
                    // Make it cubic
                    double maxDim = max(max(size[0], size[1]), size[2]);
                    finalBoxVectors.push_back(Vec3(maxDim, 0, 0));
                    finalBoxVectors.push_back(Vec3(0, maxDim, 0));
                    finalBoxVectors.push_back(Vec3(0, 0, maxDim));
                } else if (boxShape == "rectangular") {
                    finalBoxVectors.push_back(Vec3(size[0], 0, 0));
                    finalBoxVectors.push_back(Vec3(0, size[1], 0));
                    finalBoxVectors.push_back(Vec3(0, 0, size[2]));
                } else {
                    // Complex shapes like dodecahedron - delegate to Python
                    return false;
                }
            }
        } else if (numAdded > 0) {
            // Calculate box size based on number of molecules
            // Rough estimate: 30 molecules per nm³ for water
            double waterDensity = 30.0; // molecules per nm³
            double volume = numAdded / waterDensity;
            double sideLength = pow(volume, 1.0/3.0);
            
            // Add some padding for the solute
            if (!impl->positions.empty()) {
                sideLength += 2.0; // Add 2 nm padding
            }
            
            finalBoxVectors.push_back(Vec3(sideLength, 0, 0));
            finalBoxVectors.push_back(Vec3(0, sideLength, 0));
            finalBoxVectors.push_back(Vec3(0, 0, sideLength));
        } else {
            // Use existing box vectors if available
            finalBoxVectors = impl->periodicBoxVectors;
            if (finalBoxVectors.size() != 3) {
                // No valid box information available
                return false;
            }
        }
    }
    
    if (finalBoxVectors.size() != 3) {
        return false; // Cannot determine valid box vectors
    }
    
    // Set the box vectors
    impl->periodicBoxVectors = finalBoxVectors;
    
    // For complex solvent addition (placing water molecules, removing overlaps, adding ions),
    // delegate to the well-tested Python implementation for now.
    // The C++ implementation has successfully set up the box, which is a key part.
    
    // TODO: Future enhancement could add:
    // - Loading pre-equilibrated water boxes
    // - Placing water molecules in a lattice
    // - Removing waters that overlap with solute atoms  
    // - Adding appropriate ions for neutralization and ionic strength
    
    return false; // Indicates to fall back to Python for molecular placement
}