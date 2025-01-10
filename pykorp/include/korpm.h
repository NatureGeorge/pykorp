#ifndef INCLUDE_KORPM_H_
#define INCLUDE_KORPM_H_

// Unique Residue (aminoacid) identifiers
short int const ALA =  0 ;
short int const CYS =  1 ;
short int const ASP =  2 ;
short int const GLU =  3 ;
short int const PHE =  4 ;
short int const GLY =  5 ;
short int const HIS =  6 ;
short int const ILE =  7 ;
short int const LYS =  8 ;
short int const LEU =  9 ;
short int const MET =  10 ;
short int const ASN =  11 ;
short int const PRO =  12 ;
short int const GLN =  13 ;
short int const ARG =  14 ;
short int const SER =  15 ;
short int const THR =  16 ;
short int const VAL =  17 ;
short int const TRP =  18 ;
short int const TYR =  19 ;

char aasAA[]={ALA, CYS, ASP, GLU, PHE, GLY, HIS, ILE, LYS, LEU, MET, ASN, PRO, GLN, ARG, SER, THR, VAL, TRP, TYR};
char frasAA[]={ALA, CYS, ASP, GLU, PHE, GLY, HIS, ILE, LYS, LEU, MET, ASN, PRO, GLN, ARG, SER, THR, VAL, TRP, TYR};

// OUR RESIDUE TYPES (from ResIni.h)
//	ALA =  0 ;
//	CYS =  1 ;
//	ASP =  2 ;
//	GLU =  3 ;
//	PHE =  4 ;
//	GLY =  5 ;
//	HIS =  6 ;
//	ILE =  7 ;
//	LYS =  8 ;
//	LEU =  9 ;
//	MET =  10 ;
//	ASN =  11 ;
//	PRO =  12 ;
//	GLN =  13 ;
//	ARG =  14 ;
//	SER =  15 ;
//	THR =  16 ;
//	VAL =  17 ;
//	TRP =  18 ;
//	TYR =  19 ;
//	ASH =  20 ;       //Neutral ASP
//	CYX =  21 ;       //SS-bonded CYS
//	CYM =  22 ;       //Negative CYS
//	GLH =  23 ;       //Neutral GLU
//	HIP =  24 ;       //Positive HIS
//	HID =  25 ;       //Neutral HIS, proton HD1 present
//	HIE =  26 ;       //Neutral HIS, proton HE2 present
//	LYN =  27 ;       //Neutral LYS
//	TYM =  28 ;       //Negative TYR
//	MSE =  29 ; 		// Seleno-Methionine
//	NtE =  30 ;
//	CtE =  31 ;
//	DGUA =  32 ;
//	DADE =  33 ;
//	DCYT =  34 ;
//	DTHY =  35 ;
//	GUA =  36 ;
//	ADE =  37 ;
//	CYT =  38 ;
//	URA =  39 ;

// Interaction frames mapping:
//   mapping[][0]--> Number of interaction frames for current residue (to screen all interactions)
//   mapping[][1]--> Interaction-specific potential map index (to map different residue interactions into the required potential map)
//   mapping[][2]--> "N" atom index of the interaction frame (for orthogonal framework definition)
//   mapping[][3]--> "CA" atom index of the interaction frame (for orthogonal framework definition and pairwise distance evaluation)
//   mapping[][4]--> "C" atom index of the interaction frame (for orthogonal framework definition)

// INTERACTION FRAMES: 20 standard aminoacids model (mappingAA) using CA for distance evaluation and N,C for interaction frame definition
char a01[] =  {1,  0, 0, 1, 2}; // ALA
char a02[] =  {1,  1, 0, 1, 2}; // CYS
char a03[] =  {1,  2, 0, 1, 2}; // ASP
char a04[] =  {1,  3, 0, 1, 2}; // GLU
char a05[] =  {1,  4, 0, 1, 2}; // PHE
char a06[] =  {1,  5, 0, 1, 2}; // GLY
char a07[] =  {1,  6, 0, 1, 2}; // HIS
char a08[] =  {1,  7, 0, 1, 2}; // ILE
char a09[] =  {1,  8, 0, 1, 2}; // LYS
char a10[] =  {1,  9, 0, 1, 2}; // LEU
char a11[] =  {1, 10, 0, 1, 2}; // MET
char a12[] =  {1, 11, 0, 1, 2}; // ASN
char a13[] =  {1, 12, 0, 1, 2}; // PRO
char a14[] =  {1, 13, 0, 1, 2}; // GLN
char a15[] =  {1, 14, 0, 1, 2}; // ARG
char a16[] =  {1, 15, 0, 1, 2}; // SER
char a17[] =  {1, 16, 0, 1, 2}; // THR
char a18[] =  {1, 17, 0, 1, 2}; // VAL
char a19[] =  {1, 18, 0, 1, 2}; // TRP
char a20[] =  {1, 19, 0, 1, 2}; // TYR
// Warning! This "chapa" is strictly required since in C you can't directly initialize an array from another. E.g.:
// int *myarray[] = { {3,4,5}, {32,54,6} }; <-- NOT ALLOWED IN C
char *mappingAA[] = { a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20 };


#endif /* INCLUDE_KORPM_H_ */
