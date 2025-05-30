#ifndef OPENMM_MODELLER_H_
#define OPENMM_MODELLER_H_

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

#include "Vec3.h"
#include <vector>
#include <string>
#include <map>
#include <set>
#include "internal/windowsExport.h"

namespace OpenMM {

/**
 * Forward declarations
 */
class ModellerImpl;

/**
 * Modeller provides tools for editing molecular models, such as adding water or missing hydrogens.
 * This is a high-performance C++ implementation of molecular model building operations.
 *
 * The Modeller class maintains internal representations of molecular topology and coordinates,
 * optimized for efficient manipulation operations like adding/removing atoms, residues, and bonds.
 */
class OPENMM_EXPORT Modeller {
public:
    /**
     * Atom represents an atom in the molecular system.
     */
    struct Atom {
        std::string name;
        int element;        // Element number (1=H, 6=C, etc.), 0 for virtual sites
        int index;          // Index in the system
        int residueIndex;   // Index of parent residue
        std::string id;     // PDB atom id
        double formalCharge;
        
        Atom(const std::string& name, int element, int index, int residueIndex, 
             const std::string& id = "", double formalCharge = 0.0) :
            name(name), element(element), index(index), residueIndex(residueIndex),
            id(id), formalCharge(formalCharge) {}
    };
    
    /**
     * Residue represents a residue in the molecular system.
     */
    struct Residue {
        std::string name;
        int index;
        int chainIndex;
        std::string id;
        std::string insertionCode;
        std::vector<int> atomIndices;
        
        Residue(const std::string& name, int index, int chainIndex,
                const std::string& id = "", const std::string& insertionCode = "") :
            name(name), index(index), chainIndex(chainIndex), id(id), insertionCode(insertionCode) {}
    };
    
    /**
     * Chain represents a chain in the molecular system.
     */
    struct Chain {
        std::string id;
        int index;
        std::vector<int> residueIndices;
        
        Chain(const std::string& id, int index) : id(id), index(index) {}
    };
    
    /**
     * Bond represents a bond between two atoms.
     */
    struct Bond {
        int atom1Index;
        int atom2Index;
        int bondType;     // 0=None, 1=Single, 2=Double, 3=Triple, 4=Aromatic, 5=Amide
        double bondOrder;
        
        Bond(int atom1, int atom2, int type = 0, double order = 0.0) :
            atom1Index(atom1), atom2Index(atom2), bondType(type), bondOrder(order) {}
    };
    
    /**
     * HydrogenDefinition contains information about hydrogen atoms to be added.
     */
    struct HydrogenDefinition {
        std::string name;
        std::string parent;
        double maxph;
        std::vector<std::string> variants;
        std::string terminal;
        
        HydrogenDefinition(const std::string& name, const std::string& parent,
                         double maxph, const std::vector<std::string>& variants,
                         const std::string& terminal) :
            name(name), parent(parent), maxph(maxph), variants(variants), terminal(terminal) {}
    };
    
    /**
     * ResidueHydrogenData contains hydrogen definitions for a residue type.
     */
    struct ResidueHydrogenData {
        std::string name;
        std::vector<std::string> variants;
        std::vector<HydrogenDefinition> hydrogens;
        
        ResidueHydrogenData(const std::string& name) : name(name) {}
    };

public:
    /**
     * Create a new Modeller object.
     */
    Modeller();
    
    /**
     * Create a new Modeller object with initial topology and positions.
     * 
     * @param chains        initial chains data
     * @param residues      initial residues data  
     * @param atoms         initial atoms data
     * @param bonds         initial bonds data
     * @param positions     initial atomic positions
     * @param periodicBoxVectors periodic box vectors (can be empty)
     */
    Modeller(const std::vector<Chain>& chains,
             const std::vector<Residue>& residues,
             const std::vector<Atom>& atoms,
             const std::vector<Bond>& bonds,
             const std::vector<Vec3>& positions,
             const std::vector<Vec3>& periodicBoxVectors = std::vector<Vec3>());
    
    ~Modeller();
    
    /**
     * Get the current atomic positions.
     */
    const std::vector<Vec3>& getPositions() const;
    
    /**
     * Set the atomic positions.
     */
    void setPositions(const std::vector<Vec3>& positions);
    
    /**
     * Get the current topology data.
     */
    void getTopology(std::vector<Chain>& chains,
                     std::vector<Residue>& residues,
                     std::vector<Atom>& atoms,
                     std::vector<Bond>& bonds) const;
    
    /**
     * Get the periodic box vectors.
     */
    const std::vector<Vec3>& getPeriodicBoxVectors() const;
    
    /**
     * Set the periodic box vectors.
     */
    void setPeriodicBoxVectors(const std::vector<Vec3>& vectors);
    
    /**
     * Add atoms, residues, chains, and bonds to the model.
     * 
     * @param addChains     chains to add
     * @param addResidues   residues to add
     * @param addAtoms      atoms to add
     * @param addBonds      bonds to add
     * @param addPositions  positions of atoms to add
     */
    void add(const std::vector<Chain>& addChains,
             const std::vector<Residue>& addResidues,
             const std::vector<Atom>& addAtoms,
             const std::vector<Bond>& addBonds,
             const std::vector<Vec3>& addPositions);
    
    /**
     * Delete atoms, residues, chains, and bonds from the model.
     * 
     * @param atomsToDelete     indices of atoms to delete
     * @param residuesToDelete  indices of residues to delete  
     * @param chainsToDelete    indices of chains to delete
     * @param bondsToDelete     pairs of atom indices representing bonds to delete
     */
    void deleteItems(const std::vector<int>& atomsToDelete,
                     const std::vector<int>& residuesToDelete,
                     const std::vector<int>& chainsToDelete,
                     const std::vector<std::pair<int,int>>& bondsToDelete);
    
    /**
     * Delete all water molecules from the model.
     * Water molecules are identified by residue names "HOH" and "WAT".
     */
    void deleteWater();
    
    /**
     * Get the number of atoms in the system.
     */
    int getNumAtoms() const;
    
    /**
     * Get the number of residues in the system.
     */
    int getNumResidues() const;
    
    /**
     * Get the number of chains in the system.
     */
    int getNumChains() const;
    
    /**
     * Get the number of bonds in the system.
     */
    int getNumBonds() const;
    
    /**
     * Load hydrogen definitions from structured data.
     * This method is designed to be called from Python with pre-parsed XML data.
     * 
     * @param residueHydrogenData  hydrogen definitions for each residue type
     */
    static void loadHydrogenDefinitions(const std::map<std::string, ResidueHydrogenData>& residueHydrogenData);
    
    /**
     * Get the currently loaded hydrogen definitions.
     */
    static const std::map<std::string, ResidueHydrogenData>& getHydrogenDefinitions();

private:
    ModellerImpl* impl;
    
    // Static hydrogen definitions loaded from XML
    static std::map<std::string, ResidueHydrogenData> residueHydrogens;
    static bool hasLoadedStandardHydrogens;
};

} // namespace OpenMM

#endif /*OPENMM_MODELLER_H_*/