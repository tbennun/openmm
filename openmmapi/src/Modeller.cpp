/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *               } else if (atom.element == "N") {
            sigma = 0.33;
            epsilon = 0.7;
        }                                                                    *
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

#include "openmm/app/Modeller.h"
#include "openmm/System.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <sstream>

using namespace OpenMM;
using namespace OpenMM::app;
using namespace std;

// Static data for element masses (in amu) and common ionic radii
static const map<string, double> ELEMENT_MASSES = {
    {"H", 1.008}, {"C", 12.011}, {"N", 14.007}, {"O", 15.999},
    {"P", 30.974}, {"S", 32.065}, {"F", 18.998}, {"Cl", 35.453},
    {"Br", 79.904}, {"I", 126.904}, {"Na", 22.990}, {"K", 39.098},
    {"Li", 6.941}, {"Cs", 132.905}, {"Rb", 85.468}, {"Ca", 40.078},
    {"Mg", 24.305}, {"Zn", 65.38}, {"Fe", 55.845}
};

// Water model geometries (OH bond length, HOH angle in degrees)
static const map<Modeller::WaterModel, pair<double, double>> WATER_GEOMETRIES = {
    {Modeller::TIP3P, {0.09572, 104.52}},      // nm, degrees
    {Modeller::SPC, {0.10, 109.47}},
    {Modeller::SPCE, {0.10, 109.47}},
    {Modeller::TIP4PEW, {0.09572, 104.52}},
    {Modeller::TIP5P, {0.09572, 104.52}},
    {Modeller::SWM4NDP, {0.09572, 104.52}}
};

// Ion charges
static const map<Modeller::IonType, double> ION_CHARGES = {
    {Modeller::SODIUM, 1.0}, {Modeller::POTASSIUM, 1.0}, {Modeller::LITHIUM, 1.0},
    {Modeller::CESIUM, 1.0}, {Modeller::RUBIDIUM, 1.0}, {Modeller::CHLORIDE, -1.0},
    {Modeller::BROMIDE, -1.0}, {Modeller::IODIDE, -1.0}, {Modeller::FLUORIDE, -1.0}
};

// Ion names for residue creation
static const map<Modeller::IonType, string> ION_NAMES = {
    {Modeller::SODIUM, "Na+"}, {Modeller::POTASSIUM, "K+"}, {Modeller::LITHIUM, "Li+"},
    {Modeller::CESIUM, "Cs+"}, {Modeller::RUBIDIUM, "Rb+"}, {Modeller::CHLORIDE, "Cl-"},
    {Modeller::BROMIDE, "Br-"}, {Modeller::IODIDE, "I-"}, {Modeller::FLUORIDE, "F-"}
};

Modeller::Modeller(const vector<AtomInfo>& atoms, const vector<Vec3>& positions) 
    : atoms_(atoms), positions_(positions), hasPeriodicBox_(false) {
    if (atoms.size() != positions.size()) {
        throw OpenMMException("Number of atoms and positions must match");
    }
    
    // Initialize box vectors to zero
    for (int i = 0; i < 3; i++) {
        boxVectors_[i] = Vec3(0, 0, 0);
    }
    
    // Build residue information from atoms
    map<int, int> residueMap; // residueIndex -> residues_ index
    for (size_t i = 0; i < atoms_.size(); i++) {
        const AtomInfo& atom = atoms_[i];
        if (residueMap.find(atom.residueIndex) == residueMap.end()) {
            ResidueInfo residue(atom.residueName, atom.chainId, atom.residueIndex);
            residues_.push_back(residue);
            residueMap[atom.residueIndex] = static_cast<int>(residues_.size() - 1);
        }
        residues_[residueMap[atom.residueIndex]].atoms.push_back(static_cast<int>(i));
    }
}

Modeller::Modeller(const System& system, const vector<Vec3>& positions) 
    : positions_(positions), hasPeriodicBox_(false) {
    if (system.getNumParticles() != static_cast<int>(positions.size())) {
        throw OpenMMException("Number of particles in system and positions must match");
    }
    
    // Initialize box vectors to zero
    for (int i = 0; i < 3; i++) {
        boxVectors_[i] = Vec3(0, 0, 0);
    }
    
    // Extract basic atom information from system
    // Note: Limited topology information available from System
    atoms_.reserve(system.getNumParticles());
    for (int i = 0; i < system.getNumParticles(); i++) {
        double mass = system.getParticleMass(i);
        
        // Try to infer element from mass (very basic)
        string element = "C"; // default
        if (abs(mass - 1.008) < 0.1) element = "H";
        else if (abs(mass - 12.011) < 0.1) element = "C";
        else if (abs(mass - 14.007) < 0.1) element = "N";
        else if (abs(mass - 15.999) < 0.1) element = "O";
        else if (abs(mass - 30.974) < 0.1) element = "P";
        else if (abs(mass - 32.065) < 0.1) element = "S";
        
        AtomInfo atom("UNK", element, 0, "UNK", "A", mass, 0.0);
        atoms_.push_back(atom);
    }
    
    // Create a single residue containing all atoms
    ResidueInfo residue("UNK", "A", 0);
    for (int i = 0; i < system.getNumParticles(); i++) {
        residue.atoms.push_back(i);
    }
    residues_.push_back(residue);
}

Modeller::Modeller(const Modeller& other) 
    : atoms_(other.atoms_), positions_(other.positions_), bonds_(other.bonds_),
      residues_(other.residues_), hasPeriodicBox_(other.hasPeriodicBox_) {
    for (int i = 0; i < 3; i++) {
        boxVectors_[i] = other.boxVectors_[i];
    }
}

Modeller& Modeller::operator=(const Modeller& other) {
    if (this != &other) {
        atoms_ = other.atoms_;
        positions_ = other.positions_;
        bonds_ = other.bonds_;
        residues_ = other.residues_;
        hasPeriodicBox_ = other.hasPeriodicBox_;
        for (int i = 0; i < 3; i++) {
            boxVectors_[i] = other.boxVectors_[i];
        }
    }
    return *this;
}

Modeller::~Modeller() {
    // Default destructor is sufficient for this class
}

void Modeller::setPeriodicBoxVectors(const Vec3& a, const Vec3& b, const Vec3& c) {
    boxVectors_[0] = a;
    boxVectors_[1] = b;
    boxVectors_[2] = c;
    hasPeriodicBox_ = true;
}

bool Modeller::getPeriodicBoxVectors(Vec3& a, Vec3& b, Vec3& c) const {
    if (hasPeriodicBox_) {
        a = boxVectors_[0];
        b = boxVectors_[1];
        c = boxVectors_[2];
        return true;
    }
    return false;
}

void Modeller::add(const vector<AtomInfo>& atoms, const vector<Vec3>& positions) {
    if (atoms.size() != positions.size()) {
        throw OpenMMException("Number of atoms and positions must match");
    }
    
    int oldNumAtoms = static_cast<int>(atoms_.size());
    
    // Add atoms and positions
    atoms_.insert(atoms_.end(), atoms.begin(), atoms.end());
    positions_.insert(positions_.end(), positions.begin(), positions.end());
    
    // Update residue information
    map<int, int> residueMap;
    for (size_t i = 0; i < residues_.size(); i++) {
        residueMap[residues_[i].index] = static_cast<int>(i);
    }
    
    for (size_t i = 0; i < atoms.size(); i++) {
        const AtomInfo& atom = atoms[i];
        if (residueMap.find(atom.residueIndex) == residueMap.end()) {
            ResidueInfo residue(atom.residueName, atom.chainId, atom.residueIndex);
            residues_.push_back(residue);
            residueMap[atom.residueIndex] = static_cast<int>(residues_.size() - 1);
        }
        residues_[residueMap[atom.residueIndex]].atoms.push_back(oldNumAtoms + static_cast<int>(i));
    }
}

void Modeller::deleteAtoms(const vector<int>& atomIndices) {
    if (atomIndices.empty()) return;
    
    // Sort indices in descending order for safe deletion
    vector<int> sortedIndices = atomIndices;
    sort(sortedIndices.rbegin(), sortedIndices.rend());
    
    // Check for valid indices
    for (int index : sortedIndices) {
        if (index < 0 || index >= static_cast<int>(atoms_.size())) {
            throw OpenMMException("Invalid atom index for deletion");
        }
    }
    
    // Delete atoms and positions (in reverse order)
    for (int index : sortedIndices) {
        atoms_.erase(atoms_.begin() + index);
        positions_.erase(positions_.begin() + index);
    }
    
    // Update bonds - remove bonds involving deleted atoms and update indices
    set<int> deletedSet(atomIndices.begin(), atomIndices.end());
    vector<BondInfo> newBonds;
    
    for (const BondInfo& bond : bonds_) {
        if (deletedSet.find(bond.atom1) == deletedSet.end() && 
            deletedSet.find(bond.atom2) == deletedSet.end()) {
            
            // Update indices to account for deleted atoms
            int newAtom1 = bond.atom1;
            int newAtom2 = bond.atom2;
            
            for (int deletedIndex : sortedIndices) {
                if (deletedIndex < bond.atom1) newAtom1--;
                if (deletedIndex < bond.atom2) newAtom2--;
            }
            
            newBonds.push_back(BondInfo(newAtom1, newAtom2, bond.order));
        }
    }
    bonds_ = newBonds;
    
    // Update residue atom lists
    for (ResidueInfo& residue : residues_) {
        vector<int> newAtoms;
        for (int atomIndex : residue.atoms) {
            if (deletedSet.find(atomIndex) == deletedSet.end()) {
                // Update index to account for deleted atoms
                int newIndex = atomIndex;
                for (int deletedIndex : sortedIndices) {
                    if (deletedIndex < atomIndex) newIndex--;
                }
                newAtoms.push_back(newIndex);
            }
        }
        residue.atoms = newAtoms;
    }
    
    // Remove empty residues
    residues_.erase(
        remove_if(residues_.begin(), residues_.end(),
                 [](const ResidueInfo& res) { return res.atoms.empty(); }),
        residues_.end());
}

void Modeller::addHydrogens(const ForceFieldInfo& forcefield, double pH,
                           const map<int, map<string, string>>& variants,
                           Platform* platform, bool minimizeEnergy) {
    
    vector<AtomInfo> newAtoms;
    vector<Vec3> newPositions;
    vector<int> newHydrogenIndices;
    
    // Process each residue for hydrogen addition
    for (const ResidueInfo& residue : residues_) {
        for (int atomIndex : residue.atoms) {
            const AtomInfo& atom = atoms_[atomIndex];
            
            // Skip if already hydrogen
            if (atom.element == "H") continue;
            
            // Determine if this atom needs hydrogens based on element and bonding
            vector<int> neighbors = findBondedAtoms(atomIndex);
            int expectedBonds = getExpectedBondCount(atom.element);
            int currentBonds = static_cast<int>(neighbors.size());
            
            if (currentBonds < expectedBonds) {
                int hydrogensToAdd = expectedBonds - currentBonds;
                
                // Use variant information if provided
                string variant = "";
                auto resIt = variants.find(residue.index);
                if (resIt != variants.end()) {
                    auto atomIt = resIt->second.find(atom.name);
                    if (atomIt != resIt->second.end()) {
                        variant = atomIt->second;
                    }
                }
                
                if (variant.empty()) {
                    variant = selectHydrogenVariant(residue.index, atom.name, pH, forcefield);
                }
                
                // Add hydrogens based on variant and geometry
                for (int h = 0; h < hydrogensToAdd; h++) {
                    Vec3 hPos = calculateHydrogenPosition(atomIndex, neighbors, forcefield);
                    
                    string hName = atom.name + "H";
                    if (hydrogensToAdd > 1) {
                        hName += to_string(h + 1);
                    }
                    
                    AtomInfo hydrogen(hName, "H", atom.residueIndex, atom.residueName, 
                                    atom.chainId, 1.008, 0.0);
                    
                    newAtoms.push_back(hydrogen);
                    newPositions.push_back(hPos);
                    newHydrogenIndices.push_back(static_cast<int>(atoms_.size() + newAtoms.size() - 1));
                    
                    // Add bond to heavy atom
                    bonds_.push_back(BondInfo(atomIndex, static_cast<int>(atoms_.size() + newAtoms.size() - 1), 1));
                }
            }
        }
    }
    
    // Add the new atoms and positions
    add(newAtoms, newPositions);
    
    // Optimize hydrogen positions if requested
    if (minimizeEnergy && !newHydrogenIndices.empty()) {
        optimizeHydrogenPositions(newHydrogenIndices, forcefield);
    }
}

void Modeller::addSolvent(WaterModel model, const Vec3* boxSize, double padding,
                         double ionicStrength, IonType positiveIon, IonType negativeIon,
                         bool neutralize) {
    
    Vec3 solventBox;
    if (boxSize != nullptr) {
        solventBox = *boxSize;
    } else {
        // Calculate automatic box size based on solute dimensions plus padding
        Vec3 minCoord(1e6, 1e6, 1e6);
        Vec3 maxCoord(-1e6, -1e6, -1e6);
        
        for (const Vec3& pos : positions_) {
            minCoord[0] = min(minCoord[0], pos[0]);
            minCoord[1] = min(minCoord[1], pos[1]);
            minCoord[2] = min(minCoord[2], pos[2]);
            maxCoord[0] = max(maxCoord[0], pos[0]);
            maxCoord[1] = max(maxCoord[1], pos[1]);
            maxCoord[2] = max(maxCoord[2], pos[2]);
        }
        
        solventBox = Vec3(maxCoord[0] - minCoord[0] + 2 * padding,
                         maxCoord[1] - minCoord[1] + 2 * padding,
                         maxCoord[2] - minCoord[2] + 2 * padding);
    }
    
    // Set periodic box vectors
    setPeriodicBoxVectors(Vec3(solventBox[0], 0, 0), 
                         Vec3(0, solventBox[1], 0), 
                         Vec3(0, 0, solventBox[2]));
    
    // Generate water positions with appropriate spacing
    double waterSpacing = 0.31; // nm, approximate distance between water molecules
    vector<Vec3> waterPositions = generateSolventPositions(solventBox, waterSpacing);
    
    // Add water molecules
    vector<AtomInfo> waterAtoms;
    vector<Vec3> allPositions;
    
    int waterResIndex = residues_.empty() ? 0 : residues_.back().index + 1;
    
    for (const Vec3& waterPos : waterPositions) {
        if (!checkOverlap(waterPos, 0.23)) { // 2.3 Å minimum distance
            addWaterMolecule(waterPos, model);
        }
    }
    
    // Add ions if requested
    if (neutralize || ionicStrength > 0.0) {
        addIons(positiveIon, negativeIon, ionicStrength, neutralize);
    }
}

void Modeller::addIons(IonType positiveIon, IonType negativeIon, double ionicStrength,
                      bool neutralize, const vector<int>& replacementWaters) {
    
    // Calculate system charge for neutralization
    double totalCharge = getTotalCharge();
    
    // Calculate number of ions needed
    int numPositiveIons = 0;
    int numNegativeIons = 0;
    
    if (neutralize) {
        if (totalCharge > 0) {
            numNegativeIons = static_cast<int>(ceil(abs(totalCharge)));
        } else if (totalCharge < 0) {
            numPositiveIons = static_cast<int>(ceil(abs(totalCharge)));
        }
    }
    
    // Add ions for ionic strength
    if (ionicStrength > 0.0) {
        // Calculate box volume in liters
        Vec3 a, b, c;
        if (getPeriodicBoxVectors(a, b, c)) {
            double volume = abs(a[0] * (b[1] * c[2] - b[2] * c[1]) +
                               a[1] * (b[2] * c[0] - b[0] * c[2]) +
                               a[2] * (b[0] * c[1] - b[1] * c[0]));
            volume *= 1e-27; // nm³ to L
            
            // Calculate additional ions needed for ionic strength
            double avogadro = 6.022e23;
            int additionalIons = static_cast<int>(ionicStrength * volume * avogadro);
            numPositiveIons += additionalIons;
            numNegativeIons += additionalIons;
        }
    }
    
    // Find water molecules to replace (use provided list or find randomly)
    vector<int> watersToReplace = replacementWaters;
    if (watersToReplace.empty()) {
        // Find water molecules automatically
        vector<int> waterIndices;
        for (size_t i = 0; i < residues_.size(); i++) {
            if (residues_[i].name == "HOH" || residues_[i].name == "WAT") {
                waterIndices.insert(waterIndices.end(), 
                                  residues_[i].atoms.begin(), residues_[i].atoms.end());
            }
        }
        
        // Randomly select waters to replace
        random_device rd;
        mt19937 gen(rd());
        shuffle(waterIndices.begin(), waterIndices.end(), gen);
        
        int totalIonsNeeded = numPositiveIons + numNegativeIons;
        watersToReplace.assign(waterIndices.begin(), 
                              waterIndices.begin() + min(totalIonsNeeded, static_cast<int>(waterIndices.size())));
    }
    
    // Replace waters with ions
    int ionIndex = 0;
    for (int waterIndex : watersToReplace) {
        if (ionIndex >= numPositiveIons + numNegativeIons) break;
        
        Vec3 ionPos = positions_[waterIndex];
        IonType ionType = (ionIndex < numPositiveIons) ? positiveIon : negativeIon;
        
        // Replace water with ion
        atoms_[waterIndex] = AtomInfo(ION_NAMES.at(ionType), 
                                     ION_NAMES.at(ionType).substr(0, ION_NAMES.at(ionType).size()-1), 
                                     atoms_[waterIndex].residueIndex,
                                     ION_NAMES.at(ionType), 
                                     atoms_[waterIndex].chainId,
                                     getElementMass(ION_NAMES.at(ionType).substr(0, ION_NAMES.at(ionType).size()-1)),
                                     ION_CHARGES.at(ionType));
        
        ionIndex++;
    }
}

void Modeller::addMembrane(const string& lipidType, const Vec3& membraneSize, double padding) {
    // Placeholder implementation for future membrane functionality
    throw OpenMMException("Membrane addition not yet implemented");
}

void Modeller::deleteWater(const vector<int>& waterIndices) {
    vector<int> atomsToDelete;
    
    if (waterIndices.empty()) {
        // Delete all water molecules
        for (const ResidueInfo& residue : residues_) {
            if (residue.name == "HOH" || residue.name == "WAT") {
                atomsToDelete.insert(atomsToDelete.end(), 
                                   residue.atoms.begin(), residue.atoms.end());
            }
        }
    } else {
        // Delete specific water molecules
        for (int waterIndex : waterIndices) {
            if (waterIndex >= 0 && waterIndex < static_cast<int>(residues_.size())) {
                const ResidueInfo& residue = residues_[waterIndex];
                if (residue.name == "HOH" || residue.name == "WAT") {
                    atomsToDelete.insert(atomsToDelete.end(), 
                                       residue.atoms.begin(), residue.atoms.end());
                }
            }
        }
    }
    
    deleteAtoms(atomsToDelete);
}

double Modeller::getTotalCharge() const {
    double totalCharge = 0.0;
    for (const AtomInfo& atom : atoms_) {
        totalCharge += atom.charge;
    }
    return totalCharge;
}

unique_ptr<System> Modeller::createSystem(const ForceFieldInfo& forcefield,
                                        const map<string, int>* constraints) const {
    auto system = std::unique_ptr<System>(new System());
    
    // Add particles
    for (const AtomInfo& atom : atoms_) {
        double mass = atom.mass;
        if (mass == 0.0) {
            mass = getElementMass(atom.element);
        }
        system->addParticle(mass);
    }
    
    // Set periodic box vectors if defined
    if (hasPeriodicBox_) {
        system->setDefaultPeriodicBoxVectors(boxVectors_[0], boxVectors_[1], boxVectors_[2]);
    }
    
    // Add forces (basic implementation)
    if (!bonds_.empty()) {
        auto bondForce = new HarmonicBondForce();
        for (const BondInfo& bond : bonds_) {
            // Use default bond parameters - in a real implementation,
            // these would come from the force field
            double k = 418400.0; // kJ/mol/nm^2 (typical C-C bond)
            double r0 = 0.154;   // nm (typical C-C bond length)
            bondForce->addBond(bond.atom1, bond.atom2, r0, k);
        }
        system->addForce(bondForce);
    }
    
    // Add nonbonded force for electrostatics and van der Waals
    auto nonbondedForce = new NonbondedForce();
    for (const AtomInfo& atom : atoms_) {
        double charge = atom.charge;
        double sigma = 0.35;   // nm, default LJ sigma
        double epsilon = 0.5;  // kJ/mol, default LJ epsilon
        
        // Adjust parameters based on element
        if (atom.element == "H") {
            sigma = 0.25;
            epsilon = 0.1;
        } else if (atom.element == "O") {
            sigma = 0.31;
            epsilon = 0.8;
        } else if (atom.element == "N") {
            sigma = 0.33;
            epsilon = 0.7;
        }
        
        nonbondedForce->addParticle(charge, sigma, epsilon);
    }
    
    if (hasPeriodicBox_) {
        nonbondedForce->setNonbondedMethod(NonbondedForce::PME);
        nonbondedForce->setCutoffDistance(1.0); // nm
    }
    
    system->addForce(nonbondedForce);
    
    return system;
}

Modeller::ForceFieldInfo Modeller::createBasicForceField() {
    ForceFieldInfo ff;
    
    // Basic element masses
    ff.atomMasses = ELEMENT_MASSES;
    
    // Basic atom type charges (simplified)
    ff.atomCharges["C"] = 0.0;
    ff.atomCharges["H"] = 0.0;
    ff.atomCharges["N"] = 0.0;
    ff.atomCharges["O"] = 0.0;
    ff.atomCharges["S"] = 0.0;
    ff.atomCharges["P"] = 0.0;
    
    // Basic atom types
    ff.atomTypes["C"] = "C";
    ff.atomTypes["H"] = "H";
    ff.atomTypes["N"] = "N";
    ff.atomTypes["O"] = "O";
    ff.atomTypes["S"] = "S";
    ff.atomTypes["P"] = "P";
    
    return ff;
}

double Modeller::getElementMass(const string& element) {
    auto it = ELEMENT_MASSES.find(element);
    if (it != ELEMENT_MASSES.end()) {
        return it->second;
    }
    return 12.0; // Default to carbon mass if unknown
}

bool Modeller::isIonizableElement(const string& element) {
    return element == "N" || element == "O" || element == "S" || element == "P";
}

// Private helper methods

void Modeller::addWaterMolecule(const Vec3& position, WaterModel model) {
    auto geometry = WATER_GEOMETRIES.at(model);
    double ohLength = geometry.first;
    double hohAngle = geometry.second * M_PI / 180.0; // Convert to radians
    
    int waterResIndex = residues_.empty() ? 0 : residues_.back().index + 1;
    
    // Create water atoms
    AtomInfo oxygen("O", "O", waterResIndex, "HOH", "W", 15.999, -0.834);
    AtomInfo hydrogen1("H1", "H", waterResIndex, "HOH", "W", 1.008, 0.417);
    AtomInfo hydrogen2("H2", "H", waterResIndex, "HOH", "W", 1.008, 0.417);
    
    // Calculate positions
    Vec3 oPos = position;
    Vec3 h1Pos = position + Vec3(ohLength, 0, 0);
    Vec3 h2Pos = position + Vec3(ohLength * cos(hohAngle), ohLength * sin(hohAngle), 0);
    
    // Add atoms
    int startIndex = static_cast<int>(atoms_.size());
    atoms_.push_back(oxygen);
    atoms_.push_back(hydrogen1);
    atoms_.push_back(hydrogen2);
    
    positions_.push_back(oPos);
    positions_.push_back(h1Pos);
    positions_.push_back(h2Pos);
    
    // Add bonds
    bonds_.push_back(BondInfo(startIndex, startIndex + 1, 1));
    bonds_.push_back(BondInfo(startIndex, startIndex + 2, 1));
    
    // Add residue
    ResidueInfo waterRes("HOH", "W", waterResIndex);
    waterRes.atoms = {startIndex, startIndex + 1, startIndex + 2};
    residues_.push_back(waterRes);
}

void Modeller::addIon(const Vec3& position, IonType ionType) {
    string ionName = ION_NAMES.at(ionType);
    string element = ionName.substr(0, ionName.size() - 1); // Remove charge
    double charge = ION_CHARGES.at(ionType);
    double mass = getElementMass(element);
    
    int ionResIndex = residues_.empty() ? 0 : residues_.back().index + 1;
    
    AtomInfo ion(ionName, element, ionResIndex, ionName, "I", mass, charge);
    
    int atomIndex = static_cast<int>(atoms_.size());
    atoms_.push_back(ion);
    positions_.push_back(position);
    
    ResidueInfo ionRes(ionName, "I", ionResIndex);
    ionRes.atoms = {atomIndex};
    residues_.push_back(ionRes);
}

vector<Vec3> Modeller::generateSolventPositions(const Vec3& boxSize, double spacing) {
    vector<Vec3> positions;
    
    int nx = static_cast<int>(boxSize[0] / spacing);
    int ny = static_cast<int>(boxSize[1] / spacing);
    int nz = static_cast<int>(boxSize[2] / spacing);
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                Vec3 pos(i * spacing + spacing/2, 
                        j * spacing + spacing/2, 
                        k * spacing + spacing/2);
                positions.push_back(pos);
            }
        }
    }
    
    return positions;
}

bool Modeller::checkOverlap(const Vec3& position, double minDistance) const {
    for (const Vec3& existingPos : positions_) {
        double dx = position[0] - existingPos[0];
        double dy = position[1] - existingPos[1];
        double dz = position[2] - existingPos[2];
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance < minDistance) {
            return true; // Overlap detected
        }
    }
    return false;
}

void Modeller::optimizeHydrogenPositions(const vector<int>& hydrogenIndices, 
                                        const ForceFieldInfo& forcefield) {
    // Create a simple system for energy minimization
    auto system = createSystem(forcefield);
    
    // Use Verlet integrator for energy minimization
    VerletIntegrator integrator(0.001); // 1 fs timestep
    
    // Create context and set positions
    Context context(*system, integrator);
    context.setPositions(positions_);
    
    // Run local energy minimization
    LocalEnergyMinimizer::minimize(context, 1e-4, 100);
    
    // Update positions
    State state = context.getState(State::Positions);
    positions_ = state.getPositions();
}

Vec3 Modeller::calculateHydrogenPosition(int heavyAtom, const vector<int>& neighbors,
                                        const ForceFieldInfo& forcefield) const {
    const Vec3& heavyPos = positions_[heavyAtom];
    const AtomInfo& heavy = atoms_[heavyAtom];
    
    // Default C-H bond length
    double bondLength = 0.109; // nm
    
    if (heavy.element == "N") bondLength = 0.101;
    else if (heavy.element == "O") bondLength = 0.096;
    else if (heavy.element == "S") bondLength = 0.134;
    
    // Calculate hydrogen position based on geometry
    if (neighbors.empty()) {
        // No neighbors, place hydrogen along x-axis
        return heavyPos + Vec3(bondLength, 0, 0);
    } else if (neighbors.size() == 1) {
        // One neighbor, place hydrogen opposite
        Vec3 neighborPos = positions_[neighbors[0]];
        Vec3 direction = heavyPos - neighborPos;
        double length = sqrt(direction.dot(direction));
        if (length > 0) {
            direction *= bondLength / length;
            return heavyPos + direction;
        }
        return heavyPos + Vec3(bondLength, 0, 0);
    } else {
        // Multiple neighbors, use tetrahedral or trigonal geometry
        Vec3 centerOfMass(0, 0, 0);
        for (int neighbor : neighbors) {
            centerOfMass += positions_[neighbor];
        }
        centerOfMass *= 1.0 / neighbors.size();
        
        Vec3 direction = heavyPos - centerOfMass;
        double length = sqrt(direction.dot(direction));
        if (length > 0) {
            direction *= bondLength / length;
            return heavyPos + direction;
        }
        return heavyPos + Vec3(bondLength, 0, 0);
    }
}

string Modeller::selectHydrogenVariant(int residueIndex, const string& atomName,
                                      double pH, const ForceFieldInfo& forcefield) const {
    // Simplified pH-dependent variant selection
    // In a real implementation, this would use the forcefield's variant rules
    
    if (atomName == "N" && pH < 6.0) {
        return "protonated";
    } else if (atomName == "O" && pH > 8.0) {
        return "deprotonated";
    }
    
    return "default";
}

vector<int> Modeller::findBondedAtoms(int atomIndex) const {
    vector<int> bonded;
    for (const BondInfo& bond : bonds_) {
        if (bond.atom1 == atomIndex) {
            bonded.push_back(bond.atom2);
        } else if (bond.atom2 == atomIndex) {
            bonded.push_back(bond.atom1);
        }
    }
    return bonded;
}

int Modeller::getExpectedBondCount(const string& element) const {
    // Typical valencies for common elements
    if (element == "H") return 1;
    if (element == "C") return 4;
    if (element == "N") return 3;
    if (element == "O") return 2;
    if (element == "S") return 2;
    if (element == "P") return 3;
    return 0; // Unknown element
}
