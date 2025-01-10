#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Necessary code from https://github.com/chaconlab/Korp/blob/main/sbg/src [BEGIN]

#include "korpe.h"
#include "korpm.h"

// Read header data into map file (common to Binned maps and GMM)
void readMapHeader(FILE *f_map, int *dimensions, float *cutoff, char *frame_model, int *nonbonding, int *nonbonding2, float *bonding_factor, int *ngauss, bool *fullgauss, bool *use_ji, bool *each_bonding)
{
	fread(dimensions, sizeof(int), 1, f_map); // Number of dimensions (3-, 4-, or 6-D)
	fread(cutoff, sizeof(float), 1, f_map); // Distance Cutoff used in map
	fread(frame_model, sizeof(char), 1, f_map); // Frame model used to obtain contacts
	fread(nonbonding, sizeof(int), 1, f_map); // Lower cutoff (nb < |i-j|)
	fread(nonbonding2, sizeof(int), 1, f_map); // Upper cutoff (|i-j| < nb2)
	fread(bonding_factor, sizeof(float), 1, f_map); // Bonding map factor

	// Mon: Add here --symNB and/or --useJI ????
	fread(ngauss, sizeof(int), 1, f_map); // Number of GMM gaussians (<=0 for Binned maps)

	char dummychar;
	fread(&dummychar, sizeof(char), 1, f_map); // Full (1) or Diagonal (0) gaussians
	if(dummychar > 0)
		*fullgauss = true; // Set either true or false for using Full (asymmetric) or Diagonal (spherical) Gaussian functions in the GMM
	else
		*fullgauss = false; // Set either true or false for using Full (asymmetric) or Diagonal (spherical) Gaussian functions in the GMM

	fread(&dummychar, sizeof(char), 1, f_map); // Use ji (1) or Not-use ji (0) interactions (only for 3D)
	if(dummychar > 0)
		*use_ji= true; // Set either true or false for using ji (1) or Not-using ji (0) interactions (only for 3D)
	else
		*use_ji = false; // Set either true or false for using ji (1) or Not-using ji (0) interactions (only for 3D)

	fread(&dummychar, sizeof(char), 1, f_map); // "Each bonding", to separate bonding contacts (i+1, i+2, etc...)
	if(dummychar > 0)
		*each_bonding = true; // Set either true or false for using ji (1) or Not-using ji (0) interactions (only for 3D)
	else
		*each_bonding = false; // Set either true or false for using ji (1) or Not-using ji (0) interactions (only for 3D)

	fread(&dummychar, sizeof(char), 1, f_map); // Some extra unused char... for future use

	int dummyint;
	fread(&dummyint, sizeof(int), 1, f_map); // Some extra unused integer... for future use
	fread(&dummyint, sizeof(int), 1, f_map); // Some extra unused integer... for future use
}

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
void initFrames(char frame_model, char ***p_mapping, int *model, int *dimensions, char **aas, char **fras, char *nft, int *nintres, int *nintfra)
{
	// Common parameters
	*aas = aasAA;
	*fras = frasAA;
	*nintres = 20;
	*nintfra = 20;

	// Specific parameters
	switch(frame_model)
	{
		case 10:  // 1-frame/aa using CA for distance-checking
			printf("Frame model %d => 1-frame/aa using CA for distances and N,C for frames\n",frame_model);
			*model = 0;
			*nft = 1; // number of frame types
			*p_mapping = mappingAA;
			break;
		default:
			printf("Please, select a valid frame model! (you've entered %d)\n Forcing exit!\n",frame_model);
			exit(2);
			break;
	}
}

// Read meshes and related variables
//  nr     --> Number of radial bins
//  br     --> Boundaries for radial bins
//  scell  --> Number of cells per shell array
mesh **readMeshes(FILE *f_map, int *nr, float **p_br, int **scell )
{
	// Getting "nr" from Map file
	fread(nr, sizeof(int), 1, f_map); // 3rd data is an integer with the Number of radial bins expected in file
	// Allocate memory for "br"
	*p_br = (float *) malloc( sizeof(float) * ((*nr) + 1) );
	// Getting "br" from Map file
	fread(*p_br, sizeof(float), *nr+1, f_map); // 4th, "nr+1" data is an array of floats with the Boundaries of radial bins
	// Computing Number of cells in each shell
	*scell = (int *) malloc( sizeof(int) * (*nr) );
	fread(*scell, sizeof(int), *nr, f_map); // 5th, "nr" data is an array of integers with the Number of cells within each shell

	// Get meshes
	mesh **meshes;
	meshes = getMeshes(f_map, *nr);
	return meshes;
}

// Get raw meshes from binary FILE (automatic memory allocation)
// INPUT: f_map --> File handle (already open)
//        nr    --> Number of shells
// OUTPUT: Meshes array read
mesh **getMeshes(FILE *f_map, int nr)
{
	mesh **meshes;
	if( !(meshes = (mesh **) malloc( sizeof(mesh *) * nr ) ) )
	{
		fprintf(stderr,"getMeshes> Memory Error in %d shells allocation, forcing exit!\n",nr);
		exit(1);
	}

	for(int i=0; i<nr; i++) // Screen shells
	{
		if( !(meshes[i] = (mesh *) malloc( sizeof(mesh) ) ) )
		{
			fprintf(stderr,"getMeshes> Memory Error in mesh structure allocation, forcing exit!\n");
			exit(1);
		}

		fread(&meshes[i]->ncells, sizeof(int), 1, f_map);                   // 1 INT --> Number of cells (total)
		fread(&meshes[i]->nring, sizeof(int), 1, f_map);                    // 2 INT --> Number of rings
		meshes[i]->ncellsring = (int *) malloc( sizeof(int) * meshes[i]->nring); // Memory allocation
		fread(meshes[i]->ncellsring, sizeof(int), meshes[i]->nring, f_map); // 3 Array of INTs --> Number of cells per ring
		meshes[i]->icell = (int *) malloc( sizeof(int) * meshes[i]->nring); // Memory allocation
		fread(meshes[i]->icell, sizeof(int), meshes[i]->nring, f_map);      // 4 Array of INTs --> Index of the first cell per ring
		meshes[i]->theta = (float *) malloc( sizeof(float) * meshes[i]->nring); // Memory allocation
		fread(meshes[i]->theta, sizeof(float), meshes[i]->nring, f_map);    // 5 Array of FLOATs --> Theta angles per ring
		meshes[i]->dpsi = (float *) malloc( sizeof(float) * meshes[i]->nring); // Memory allocation
		fread(meshes[i]->dpsi, sizeof(float), meshes[i]->nring, f_map);     // 6 Array of FLOATs -->  Psi increment per ring
		fread(&meshes[i]->dchi, sizeof(float), 1, f_map);     				// 7 FLOAT -->  Chi increment per shell
		//printf("getMeshes> %f\n", meshes[i]->dchi);
	}

	return meshes;
}

// Reads a complete KORP energy map (valid for 3/4/6D)
//  INPUT:  file --> KORP map file name
//          bonding_factor --> (optional) if >= 0.0 --> Using input bonding factor, otherwise using map's bonding factor.
//  OUTPUT: KORP map pointer (memory automatically allocated within this function)
korp *readKORP(char *file, float bonding_factor)
{
	char prog[] = "readKORP";
	FILE *f_map; // Binary potential map file (it includes Mesh info too)
	korp *map; // KORP's map data structure

	// Allocate one instance of the KORP's map data structure
	map = (korp *) malloc( sizeof(korp) * 1);

	// Open for reading a binary map
	if( !(f_map = fopen(file,"rb") ) )
	{
		printf( "%s> Error, I can't read %s binary file! Forcing exit!\n", prog, file );
		exit(1);
	}

	// Read header data into map file (common to Binned maps and GMM)
	readMapHeader(f_map, &map->dimensions, &map->cutoff, &map->frame_model, &map->nonbonding, &map->nonbonding2, &map->bonding_factor, &map->ngauss, &map->fullgauss, &map->use_ji, &map->each_bonding);

	if(bonding_factor >= 0.0)
		map->bonding_factor = bonding_factor; // Overriding bonding_factor by user input

	// maxr = cutoff; // Maximum value of Rab

	if(map->nonbonding2 >= 0) // Trick to trigger Bonding/Non-bonding energy maps
		map->use_bonding = true;
	else
		map->use_bonding = false;

	//	if(!bf_specified) // use file's bonding_factor if parser's bonding_factor was not provided
	//		map->bonding_factor = dummybf; // map's bonding factor
	//	else
	//		map->bonding_factor = bonding_factor; // MON: do this?

	// Show read parameters
	fprintf(stdout,"%s> dimensions= %d  cutoff= %.3f  Frame model %d  nb= %d  nb2= %d  use_bonding= %d  bonding_factor= %.3f  ngauss= %d  fullgauss= %d  useJI= %d  each_bonding= %d found in %s map.\n",
			prog, map->dimensions, map->cutoff, map->frame_model, map->nonbonding, map->nonbonding2, map->use_bonding, map->bonding_factor, map->ngauss, map->fullgauss, map->use_ji, map->each_bonding, file);

	initFrames(map->frame_model, &map->mapping, &map->model, &map->dimensions, &map->aas, &map->fras, &map->nft, &map->nintres, &map->nintfra); // Frame model initialization

	// Get raw meshes from binary FILE (automatic memory allocation)
	
	printf("%s> Reading raw meshes from %s\n",prog,file);
	map->meshes = readMeshes(f_map, &map->nr, &map->br, &map->scell );
	map->ncells = map->scell[map->nr-1]; // Number of cells in the "cutoff" shell
	
	// the other parameters have been read above...
	map->minr = map->br[0]; // Set "minr"

	// Getting "nchis" form Map file
	printf("%s> Current potential map is %dD and frame model %d (parser input overridden by map)\n",prog,map->dimensions,map->frame_model);

	// Initialize 1-letter code into ASCII-code indices (fast trick ;-)
	map->iaa = (char *) malloc( sizeof(char) * 256);
	for(int i=0; i<256; i++)
		map->iaa[i] = -1; // Initialization (if iaa[i] == -1, then the residue is not mapped)
	for(int i=0; i<map->nintres; i++)
		map->iaa[(int)map->aas[i] ] = i;

	// MON: nslices can be determined first and then allocate memory... consider review: Warning in 3D...further check...
	map->nslices = 10;
	map->fmapping = (float *) malloc( sizeof(float) * map->nslices ); // = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}; // Sequential distance bonding factor. It returns the "bonding factor" for the corresponding Bonding (or Non-Bonding) interaction
	for(int x=0; x < map->nslices; x++)
		map->fmapping[x] = 0.0; // Initialization... to avoid some valgrind Warnings...

	map->fmapping[0] = 1.0; // Non-Bonding always 1.0 (as reference point)
	if(map->each_bonding) // Consider each i+nb, i+nb+1, ..., i+nb+bnb, bonding interaction individually.
	{
		map->nslices = 1 + map->nonbonding2 - map->nonbonding; // |i-j| > nb to be considered, and |i-j| > bnb for non-bonding
		for(int x=1; x<map->nslices; x++)
			map->fmapping[x] = map->bonding_factor;
	}
	else // All i+nb, i+nb+1, ..., i+nb+bnb, bonding interactions are considered at the same time.
	{
		if(map->use_bonding)
		{
			map->nslices = 2; // 0= non-bonding and 1= bonding
			map->fmapping[1] = map->bonding_factor; // Bonding factor
		}
		else
			map->nslices = 1; // 0= non-bonding
	}

	map->nsmaps = map->nslices; // nsmaps is required for bonding/nonbonding handling...

	// Sequential Mapping (for bonding/non-bonding stuff)
	map->smapping = (char *) malloc( sizeof(char) * 10 );
	for(int x=0; x<map->nslices; x++)
		map->smapping[x] = 0; // Initialization... to avoid some valgrind Warnings...

	map->smapping[0] = 0; // Non-Bonding always go at "0" position
	fprintf(stdout,"%s> S-mapping: %d",prog,map->smapping[0]);
	int bi=1; // bonding index
	for(int i=1; i<10; i++)
	{
		if(i<=map->nonbonding)
			map->smapping[i] = -1; // Not-considered
		else
			if(i>map->nonbonding2 || !map->use_bonding)
				map->smapping[i] = 0; // Non-bonding
			else // Bonding
			{
				if(map->each_bonding)
				{
					map->smapping[i] = bi; // Bonding i+1, i+2, etc...
					bi++; // increase bonding index for "slice" mapping
				}
				else
					map->smapping[i] = 1; // Bonding (one map for all bonding)
			}
		fprintf(stdout," %d",map->smapping[i]);
	}
	fprintf(stdout,"\n");
	// This should change for considering i+1, i+2, etc... independent maps

	fprintf(stdout,"%s> F-mapping: ",prog);
	for(int i=0; i<10; i++)
		fprintf(stdout," %5.2f",map->fmapping[i]);
	fprintf(stdout,"\n");

	map->maps = (float *********) malloc( sizeof(float ********) * map->nslices);
	int smap = 0; // Size of Map [bytes]

	int nmaps; // Total number of individual potential maps
	//int imaps = 0; // Current map index
	nmaps = map->nslices * map->nft * map->nft * map->nintfra * map->nintfra; // HINT: nslices = 2; nft = 1; nintfra = 20;

	printf("%s> Reading Bin-wise %dD maps (total maps: %d):\n",prog,map->dimensions,nmaps);
	for(int s=0; s<map->nslices; s++)
	{
			map->maps[s] = NULL;
			initMap8D(&map->maps[s],map->nft); // allocate and initialize the first 2D (xy)

			smap=0;
			for(int x=0; x<map->nft; x++)
				for(int y=0; y<map->nft; y++) // AP and PA
				{
					initMap6D(&map->maps[s][x][y],map->nintfra); // allocate and initialize the first 2D (ij)

					// char iname[5],jname[5];

					for(int i=0; i<map->nintfra; i++)
					{
						// Skip missing intereacting frames, e.g. GLY does not have CB...
						if(x >= map->mapping[i][0]) // number of interacting frames for i-th residue
							continue;

						// resname_from_resnum(aas[i],iname);

						for(int j=0; j<map->nintfra; j++)
						{
							// Skip missing intereacting frames, e.g. GLY does not have CB...
							if(y >= map->mapping[j][0]) // number of interacting frames for j-th residue
								continue;

							// resname_from_resnum(aas[j],jname);
							// printf("%s> Reading Bin-wise %dD map (%s-%s): %1d %2d %2d %2d %2d\n",prog,dimensions,iname,jname,s,x,y,i,j);

							// fprintf(stdout,"%s> Reading 6D-map %c-%c (i-j and j-i)\n",prog,aas[i],aas[j]);
							// 6D
							map->maps[s][x][y][i][j] = getMap(f_map, map->meshes, map->nr, &smap); // Get raw 6D Map from binary FILE (automatic memory allocation)
						}
					}
				}
	}
	
	printf("%s> Map reading stuff finished!\n",prog);

	return map;
}

// 8D Float Map initialization: 20x20 (2D) + r(1D) + thetaA,psiA(2D) + thetaA,psiA(2D) + chi(1D)
void initMap6D(float  *******p_map, int side)
{
	float ******aamaps; // 6D Maps for the 210 non-redundant interactions (each aamaps[i][j] will contain a complete map)
	aamaps = (float ******) malloc( sizeof(float *****) * side );
	for(int i=0; i<side; i++)
	{
		aamaps[i] = (float *****) malloc( sizeof(float ****) * side );
		for(int j=0; j<side; j++)
			aamaps[i][j] = NULL;
	}
	*p_map = aamaps; // output
}

// 10D Float Map initialization: xy-frame-type + 20x20 (2D) + r(1D) + thetaA,psiA(2D) + thetaA,psiA(2D) + chi(1D)
void initMap8D(float  *********p_map, int side)
{
	float ********aamaps; // 6D Maps for the 210 non-redundant interactions (each aamaps[i][j] will contain a complete map)
	aamaps = (float ********) malloc( sizeof(float *******) * side );
	for(int i=0; i<side; i++)
	{
		aamaps[i] = (float *******) malloc( sizeof(float ******) * side );
		for(int j=0; j<side; j++)
			aamaps[i][j] = NULL;
	}
	*p_map = aamaps; // output
}

// Get raw Map from binary FILE handle (automatic memory allocation)
// INPUT: f_map --> File handle (already open)
//        meshes --> Meshes array
//        nr    --> Number of shells
// OUTPUT: p_smap --> Size of map [bytes]
//         return --> Map read
float ****getMap(FILE *f_map, mesh **meshes, int nr, int *p_smap)
{
	float ****map;
	int smap=0; // size of map [bytes]
	int nc;

	map = (float ****) malloc( sizeof(float ***) * nr); // allocate radial dimension (R)
	smap += sizeof(float ***) * nr;

	for(int r=0; r<nr; r++) // Screen meshes (shells)
	{
		nc = (int) roundf( 2*M_PI / meshes[r]->dchi );

		map[r] = (float ***) malloc( sizeof(float **) * meshes[r]->ncells ); // Allocate dimensions (ThetaA,PsiA) for A residue
		smap += sizeof(float **) * meshes[r]->ncells;

		for(int a=0; a<meshes[r]->ncells; a++)
		{
			map[r][a] = (float **) malloc( sizeof(float *) * meshes[r]->ncells ); // Allocate dimensions (ThetaB,PsiB) for B residue
			smap += sizeof(float *) * meshes[r]->ncells;
			for(int b=0; b<meshes[r]->ncells; b++)
			{
				map[r][a][b] = (float *) malloc( sizeof(float) * nc ); // Allocate cells for Chis
				smap += sizeof(float) * nc;
				for(int c=0; c<nc; c++)
				{
					fread(&map[r][a][b][c], sizeof(float), 1, f_map); // Read all orientation bins for A residue
					// fprintf(stderr,"r= %d  a= %d  b= %d  c= %d\n",r,a,b,c);
				}
			}
		}
	}

	*p_smap += smap;
	return map;
}

// Necessary code from https://github.com/chaconlab/Korp/blob/main/sbg/src [END]

namespace py = pybind11;

py::dict korp_to_dict(const korp* map) {
    py::dict result;

    result["model"] = map->model;
    result["dimensions"] = map->dimensions;
    result["cutoff"] = map->cutoff;
    result["frame_model"] = map->frame_model;
    result["nonbonding"] = map->nonbonding;
    result["nonbonding2"] = map->nonbonding2;
    result["bonding_factor"] = map->bonding_factor;
    result["ngauss"] = map->ngauss;
    result["fullgauss"] = map->fullgauss;
    result["use_ji"] = map->use_ji;
    result["each_bonding"] = map->each_bonding;
    result["use_bonding"] = map->use_bonding;
    result["minr"] = map->minr;
    result["nr"] = map->nr;
    result["ncells"] = map->ncells;
    result["nslices"] = map->nslices;
    result["nsmaps"] = map->nsmaps;
    result["nft"] = map->nft;
    result["nintres"] = map->nintres;
    result["nintfra"] = map->nintfra;
    
    result["br"] = py::array_t<float>({map->nr + 1}, map->br);
    result["scell"] = py::array_t<int>({map->nr}, map->scell);
    result["iaa"] = py::array_t<char>({256}, map->iaa);
    result["fmapping"] = py::array_t<float>({map->nslices}, map->fmapping);
    result["smapping"] = py::array_t<char>({10}, map->smapping);
    result["aas"] = py::array_t<char>({map->nintres}, map->aas);
    result["fras"] = py::array_t<char>({map->nintfra}, map->fras);
    
    py::array_t<char> mapping_array({20, 5});
    auto mapping_buf = mapping_array.mutable_unchecked<2>();
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 5; ++j) {
            mapping_buf(i, j) = map->mapping[i][j];
        }
    }
    result["mapping"] = mapping_array;

    py::list meshes;
    for (int i = 0; i < map->nr; ++i) {
        py::dict mesh;
        mesh["ncells"] = map->meshes[i]->ncells;
        mesh["nring"] = map->meshes[i]->nring;
        mesh["ncellsring"] = py::array_t<int>({map->meshes[i]->nring}, map->meshes[i]->ncellsring);
        mesh["icell"] = py::array_t<int>({map->meshes[i]->nring}, map->meshes[i]->icell);
        mesh["theta"] = py::array_t<float>({map->meshes[i]->nring}, map->meshes[i]->theta);
        mesh["dpsi"] = py::array_t<float>({map->meshes[i]->nring}, map->meshes[i]->dpsi);
        mesh["dchi"] = map->meshes[i]->dchi;
        meshes.append(mesh);
    }
    result["meshes"] = meshes;

    std::vector<size_t> maps_shape = {
        static_cast<size_t>(map->nslices),
        static_cast<size_t>(map->nft),
        static_cast<size_t>(map->nft),
        static_cast<size_t>(map->nintfra),
        static_cast<size_t>(map->nintfra),
        static_cast<size_t>(map->nr),
        static_cast<size_t>(map->meshes[0]->ncells),
        static_cast<size_t>(map->meshes[0]->ncells),
        static_cast<size_t>(roundf(2 * M_PI / map->meshes[0]->dchi))
    };
    py::array_t<float> maps_data(maps_shape);
    auto maps_buf = maps_data.mutable_unchecked<9>();
    for (int s = 0; s < map->nslices; ++s) {
        for (int x = 0; x < map->nft; ++x) {
            for (int y = 0; y < map->nft; ++y) {
                for (int i = 0; i < map->nintfra; ++i) {
                    for (int j = 0; j < map->nintfra; ++j) {
                        for (int r = 0; r < map->nr; ++r) {
                            for (int a = 0; a < map->meshes[r]->ncells; ++a) {
                                for (int b = 0; b < map->meshes[r]->ncells; ++b) {
                                    int nc = static_cast<int>(roundf(2 * M_PI / map->meshes[r]->dchi));
                                    for (int c = 0; c < nc; ++c) {
                                        maps_buf(s, x, y, i, j, r, a, b, c) = map->maps[s][x][y][i][j][r][a][b][c];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    result["maps"] = maps_data;

    return result;
}

PYBIND11_MODULE(c_korp, m) {
    m.def("read_korp", [](const std::string& file, float bonding_factor) {
        korp* map = readKORP(const_cast<char*>(file.c_str()), bonding_factor);
        return korp_to_dict(map);
    }, py::arg("file"), py::arg("bonding_factor") = -1.0f);
}
