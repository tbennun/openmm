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
 * Portions copyright (c) 2024-2025 Stanford University and the Authors.      *
 * Authors: OpenMM Contributors                                               *
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

#include "openmm/Vec3.h"
#include "openmm/System.h"
#include "openmm/internal/windowsExport.h"
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace OpenMM {

class Platform;
class Force;
class Context;

namespace app {

/**
 * Modeller provides tools for editing molecular systems, such as adding atoms, 
 * removing atoms, adding solvent, and adding membrane. It is designed to provide
 * high-performance C++ implementations of molecular system preparation operations
 * that are commonly performed before running molecular dynamics simulations.
 *
 * The primary use cases include:
 * - Adding hydrogen atoms to a molecular system with proper protonation states
 * - Adding explicit solvent (water) molecules around a solute
 * - Adding ions for neutralization and ionic strength control
 * - Adding membrane lipids for membrane protein simulations
 * - Basic molecular editing operations (add/delete atoms)
 *
 * This class works with simplified topology representations and particle position
 * vectors, making it suitable for high-performance molecular preparation workflows.
 */
class OPENMM_EXPORT Modeller {
public:
    /**
     * Water model types supported for solvation
     */
    enum WaterModel {
        TIP3P,      ///< TIP3P water model
        SPC,        ///< SPC water model  
        SPCE,       ///< SPC/E water model
        TIP4PEW,    ///< TIP4P-Ew water model
        TIP5P,      ///< TIP5P water model
        SWM4NDP     ///< SWM4-NDP water model
    };

    /**
     * Ion types supported for neutralization and ionic strength
     */
    enum IonType {
        SODIUM,     ///< Na+ ion
        POTASSIUM,  ///< K+ ion
        LITHIUM,    ///< Li+ ion
        CESIUM,     ///< Cs+ ion
        RUBIDIUM,   ///< Rb+ ion
        CHLORIDE,   ///< Cl- ion
        BROMIDE,    ///< Br- ion
        IODIDE,     ///< I- ion
        FLUORIDE    ///< F- ion
    };

    /**
     * Atom information structure for simplified topology representation
     */
    struct AtomInfo {
        std::string name;           ///< Atom name
        std::string element;        ///< Element symbol
        int residueIndex;           ///< Index of containing residue
        std::string residueName;    ///< Name of containing residue
        std::string chainId;        ///< Chain identifier
        double mass;                ///< Atomic mass in amu
        double charge;              ///< Partial charge in electron units
        
        AtomInfo() : residueIndex(-1), mass(0.0), charge(0.0) {}
        AtomInfo(const std::string& name, const std::string& element, 
                int residueIndex, const std::string& residueName, 
                const std::string& chainId = "", double mass = 0.0, double charge = 0.0)
            : name(name), element(element), residueIndex(residueIndex), 
              residueName(residueName), chainId(chainId), mass(mass), charge(charge) {}
    };

    /**
     * Bond information structure
     */
    struct BondInfo {
        int atom1;                  ///< Index of first atom
        int atom2;                  ///< Index of second atom
        int order;                  ///< Bond order (1=single, 2=double, etc.)
        
        BondInfo() : atom1(-1), atom2(-1), order(1) {}
        BondInfo(int atom1, int atom2, int order = 1) 
            : atom1(atom1), atom2(atom2), order(order) {}
    };

    /**
     * Residue information structure
     */
    struct ResidueInfo {
        std::string name;           ///< Residue name
        std::string chainId;        ///< Chain identifier
        int index;                  ///< Residue index/number
        std::vector<int> atoms;     ///< Indices of atoms in this residue
        
        ResidueInfo() : index(-1) {}
        ResidueInfo(const std::string& name, const std::string& chainId = "", int index = -1)
            : name(name), chainId(chainId), index(index) {}
    };

    /**
     * Force field information needed for hydrogen addition and system building
     */
    struct ForceFieldInfo {
        std::map<std::string, double> atomMasses;        ///< Element to mass mapping
        std::map<std::string, double> atomCharges;       ///< Atom type to charge mapping
        std::map<std::string, std::string> atomTypes;    ///< Atom name to type mapping
        std::map<std::string, std::vector<std::string>> hydrogenVariants; ///< pH-dependent variants
        
        ForceFieldInfo() {}
    };

private:
    // Internal data structures
    std::vector<AtomInfo> atoms_;
    std::vector<Vec3> positions_;
    std::vector<BondInfo> bonds_;
    std::vector<ResidueInfo> residues_;
    Vec3 boxVectors_[3];
    bool hasPeriodicBox_;

    // Helper methods for internal operations
    void addWaterMolecule(const Vec3& position, WaterModel model);
    void addIon(const Vec3& position, IonType ionType);
    std::vector<Vec3> generateSolventPositions(const Vec3& boxSize, double spacing);
    bool checkOverlap(const Vec3& position, double minDistance) const;
    void optimizeHydrogenPositions(const std::vector<int>& hydrogenIndices, 
                                  const ForceFieldInfo& forcefield);
    Vec3 calculateHydrogenPosition(int heavyAtom, const std::vector<int>& neighbors,
                                  const ForceFieldInfo& forcefield) const;
    std::string selectHydrogenVariant(int residueIndex, const std::string& atomName,
                                    double pH, const ForceFieldInfo& forcefield) const;
    std::vector<int> findBondedAtoms(int atomIndex) const;
    int getExpectedBondCount(const std::string& element) const;

public:
    /**
     * Create a Modeller with initial atoms and positions.
     * 
     * @param atoms      vector of atom information
     * @param positions  vector of atomic positions in nanometers
     */
    Modeller(const std::vector<AtomInfo>& atoms, const std::vector<Vec3>& positions);

    /**
     * Create a Modeller from an OpenMM System (extracts particle masses and positions).
     * Note: This constructor has limited topology information.
     * 
     * @param system     OpenMM System object
     * @param positions  vector of atomic positions in nanometers
     */
    Modeller(const System& system, const std::vector<Vec3>& positions);

    /**
     * Copy constructor
     */
    Modeller(const Modeller& other);

    /**
     * Assignment operator
     */
    Modeller& operator=(const Modeller& other);

    /**
     * Destructor
     */
    ~Modeller();

    /**
     * Get the current atoms in the system.
     * 
     * @return vector of atom information
     */
    const std::vector<AtomInfo>& getAtoms() const { return atoms_; }

    /**
     * Get the current positions of all atoms.
     * 
     * @return vector of positions in nanometers
     */
    const std::vector<Vec3>& getPositions() const { return positions_; }

    /**
     * Get the current bonds in the system.
     * 
     * @return vector of bond information
     */
    const std::vector<BondInfo>& getBonds() const { return bonds_; }

    /**
     * Get the current residues in the system.
     * 
     * @return vector of residue information
     */
    const std::vector<ResidueInfo>& getResidues() const { return residues_; }

    /**
     * Set the periodic box vectors.
     * 
     * @param a  first box vector in nanometers
     * @param b  second box vector in nanometers  
     * @param c  third box vector in nanometers
     */
    void setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c);

    /**
     * Get the periodic box vectors.
     * 
     * @param a  first box vector (output)
     * @param b  second box vector (output)
     * @param c  third box vector (output)
     * @return true if box is defined, false otherwise
     */
    bool getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const;

    /**
     * Add atoms to the system.
     * 
     * @param atoms      atoms to add
     * @param positions  positions of the new atoms in nanometers
     */
    void add(const std::vector<AtomInfo>& atoms, const std::vector<Vec3>& positions);

    /**
     * Delete atoms from the system.
     * 
     * @param atomIndices  indices of atoms to delete
     */
    void deleteAtoms(const std::vector<int>& atomIndices);

    /**
     * Add hydrogen atoms to the system based on chemical rules and pH.
     * This method uses sophisticated algorithms to determine protonation states
     * and optimal hydrogen positions.
     * 
     * @param forcefield     force field information for hydrogen placement
     * @param pH             solution pH for protonation state determination (default 7.0)
     * @param variants       optional mapping of residue->atom->variant for manual control
     * @param platform       optional platform for energy minimization (null uses reference)
     * @param minimizeEnergy whether to energy minimize hydrogen positions (default true)
     */
    void addHydrogens(const ForceFieldInfo& forcefield, 
                     double pH = 7.0,
                     const std::map<int, std::map<std::string, std::string>>& variants = {},
                     Platform* platform = nullptr,
                     bool minimizeEnergy = true);

    /**
     * Add explicit solvent molecules around the solute.
     * 
     * @param model          water model to use
     * @param boxSize        size of the solvent box (null for automatic sizing)
     * @param padding        minimum distance between solute and box edge in nm (default 1.0)
     * @param ionicStrength  ionic strength in molar (default 0.0)
     * @param positiveIon    type of positive ion for ionic strength (default SODIUM)
     * @param negativeIon    type of negative ion for ionic strength (default CHLORIDE)
     * @param neutralize     whether to add ions to neutralize the system (default true)
     */
    void addSolvent(WaterModel model,
                   const Vec3* boxSize = nullptr,
                   double padding = 1.0,
                   double ionicStrength = 0.0,
                   IonType positiveIon = SODIUM,
                   IonType negativeIon = CHLORIDE,
                   bool neutralize = true);

    /**
     * Add ions to the system for neutralization and/or ionic strength.
     * 
     * @param positiveIon    type of positive ion
     * @param negativeIon    type of negative ion
     * @param ionicStrength  target ionic strength in molar
     * @param neutralize     whether to neutralize the system charge
     * @param replacementWaters indices of water molecules to replace (empty for automatic)
     */
    void addIons(IonType positiveIon,
                IonType negativeIon, 
                double ionicStrength = 0.0,
                bool neutralize = true,
                const std::vector<int>& replacementWaters = {});

    /**
     * Add a lipid membrane to the system.
     * This is a placeholder for future membrane functionality.
     * 
     * @param lipidType      type of lipid molecules
     * @param membraneSize   size of the membrane patch
     * @param padding        padding around the membrane
     */
    void addMembrane(const std::string& lipidType,
                    const Vec3& membraneSize,
                    double padding = 1.0);

    /**
     * Remove water molecules from the system.
     * 
     * @param waterIndices  indices of water molecules to remove (empty removes all)
     */
    void deleteWater(const std::vector<int>& waterIndices = {});

    /**
     * Calculate the total charge of the system.
     * 
     * @return total charge in electron units
     */
    double getTotalCharge() const;

    /**
     * Get the number of atoms in the system.
     * 
     * @return number of atoms
     */
    int getNumAtoms() const { return static_cast<int>(atoms_.size()); }

    /**
     * Get the number of residues in the system.
     * 
     * @return number of residues  
     */
    int getNumResidues() const { return static_cast<int>(residues_.size()); }

    /**
     * Build an OpenMM System object from the current molecular system.
     * 
     * @param forcefield    force field information for system building
     * @param constraints   constraint information (null for no constraints)
     * @return OpenMM System object
     */
    std::unique_ptr<System> createSystem(const ForceFieldInfo& forcefield,
                                       const std::map<std::string, int>* constraints = nullptr) const;

    /**
     * Create a simple force field info structure with basic parameters.
     * This is a utility method for testing and simple use cases.
     * 
     * @return basic force field information
     */
    static ForceFieldInfo createBasicForceField();

    /**
     * Get the mass of an element in atomic mass units.
     * 
     * @param element  element symbol
     * @return atomic mass in amu
     */
    static double getElementMass(const std::string& element);

    /**
     * Check if an element is typically ionizable at physiological pH.
     * 
     * @param element  element symbol
     * @return true if element is typically ionizable
     */
    static bool isIonizableElement(const std::string& element);
};

} // namespace app
} // namespace OpenMM

#endif /*OPENMM_MODELLER_H_*/
