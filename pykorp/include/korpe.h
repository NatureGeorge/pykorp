#ifndef INCLUDE_KORPE_H_
#define INCLUDE_KORPE_H_

#include <stdio.h> // needed by some compilers
#include <math.h>
#include <ctype.h>

typedef struct
{
   int ncells;      // Number of constant areas (cells)
   int nring;       // Output total number of rings (including polar caps, "nring" size)
   int *ncellsring; // Number of cells per ring (including polar caps, "nring" size)
   int *icell;      // Index of the first cell of each ring from north pole (including polar caps, "nring" size)
   float *theta;    // Ring boundaries array, i.e. Latitudes (Theta) [Radians] (including polar caps, "nring" size)
   float *dpsi;     // Psi increment per ring [Radians] (including polar caps, "nring" size)
   float dchi;      // Chi increment [Radians] (constant for all rings)
} mesh;

// KORP energy map
typedef struct
{
	int model; // CG-model
	int dimensions; // Number of dimensions
	float cutoff; // Maximum distance-to-be-considered cutoff
	char frame_model; // Frame model index
	int nonbonding;
	int nonbonding2;
	float bonding_factor; // Bonding energy factor (typically 0.3)
	int ngauss;
	bool fullgauss;
	bool use_ji; // Using j-i information in 3D KORP
	bool each_bonding;
	bool use_bonding;

	char nft; // Number of frame types
	int nintres; // Number of interacting residues (to convert "aas" array into "iaa" array)
	int nintfra; // Number of interacting frames (some residues would be mapped together)
	char *aas; // Mapping identifiers to indices (WARNING, the sequential order must match "mapping" array)
	char *fras; // Frame IDs (just to dump some info)
	char **mapping; // Interaction frames mapping
	char *smapping; // smapping[10] = {0,0,0,0,0,0,0,0,0,0}; // Sequential distance Mapping. It returns the "s" index for the corresponding Bonding (or Non-Bonding) interaction

	int nr; // Number of radial bins (number of shells)
	int nchis; // Number of bins in Chi dimension
	float *br; // br[MAXSHELLS+1]; // Radial shells boundaries
	// int *ncells; // ncells[MAXSHELLS]; // Array with the number of cells per shell
	int ncells; // Number of cells in the "cutoff" shell
	int *scell; // Number of cells in each shell array
	float minr; // Minimum radial distance
	mesh **meshes; // Mesh data structures array
	char *iaa; // Given a residue unique identifier (an integer number 1 byte long) it returns the corresponding residue index in mapping
	int nslices; // Total number of interaction maps/data. Some day it will be easy adding i+1, i+2, etc. independent bonding maps.
	int nsmaps; // Number of independent maps (0= Non-bonding, 1= Bonding), when many bonding are considered it must be increased accordingly
	float *fmapping; // fmapping[10] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}; // Sequential distance bonding factor. It returns the "bonding factor" for the corresponding Bonding (or Non-Bonding) interaction

	// KORP potential map(s)
	float *******maps3D; // KORP xy-ij-3D energy Map (the full potential map)
	float *********maps; // Non-bonding KORP xy-ij-6D/4D energy Map (the full potential map)

} korp;

// Read header data into map file (common to Binned maps and GMM)
void readMapHeader(FILE *f_map, int *dimensions, float *cutoff, char *frame_model, int *nonbonding, int *nonbonding2, float *bonding_factor, int *ngauss, bool *fullgauss, bool *use_ji, bool *each_bonding);

// Initialize frames mapping for a given frame model
// INPUT:
//  frame_model  Frame model
// OUTPUT:
//  p_mapping    Frame mapping
//  model        Coarse Graining model
//  dimensions   Number of dimensions (only in 4D frame models dimensions will be updated)
//  aas          Residue indices array (for residues)
//  fras         Residue indices array (for frames)
//  nft          Number of frame types
//  nintres      Number of interacting residues (to convert "aas" array into "iaa" array)
//  nintfra      Number of interacting frames (some residues would be mapped together)
void initFrames(char frame_model, char ***p_mapping, int *model, int *dimensions, char **aas, char **fras, char *nft, int *nintres, int *nintfra);


// Read meshes and related variables
//  nr     --> Number of radial bins
//  br     --> Boundaries for radial bins
//  scell  --> Number of cells per shell array
mesh **readMeshes(FILE *f_map, int *nr, float **p_br, int **scell );


// Get raw meshes from binary FILE (automatic memory allocation)
// INPUT: f_map --> File handle (already open)
//        nr    --> Number of shells
// OUTPUT: Meshes array read
mesh **getMeshes(FILE *f_map, int nr);

// Reads a complete KORP energy map (valid for 3/4/6D)
//  INPUT:  file --> KORP map file name
//          bonding_factor --> (optional) if >= 0.0 --> Using input bonding factor, otherwise using map's bonding factor.
//  OUTPUT: KORP map pointer (memory automatically allocated within this function)
korp *readKORP(char *file, float bonding_factor = -1.0);

// 8D Integer/Float Map initialization: 20x20 (2D) + r(1D) + thetaA,psiA(2D) + thetaA,psiA(2D) + chi(1D)
void initMap6D(float *******p_map, int side);
// 10D Integer/Float Map initialization: xy-frame-type + 20x20 (2D) + r(1D) + thetaA,psiA(2D) + thetaA,psiA(2D) + chi(1D)
void initMap8D(float *********p_map, int side);

// Get raw Map from binary FILE handle (automatic memory allocation)
// INPUT: f_map --> File handle (already open)
//        meshes --> Meshes array
//        nr    --> Number of shells
// OUTPUT: p_smap --> Size of map [bytes]
//         return --> Map read
float ****getMap(FILE *f_map, mesh **meshes, int nr, int *p_smap);


#endif /* INCLUDE_KORPE_H_ */
