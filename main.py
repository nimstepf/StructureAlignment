import warnings
from typing import Union

import numpy as np

from Bio import Align
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.StructureAlignment import StructureAlignment
from Bio.SVDSuperimposer import SVDSuperimposer


def createModel(pdbfile: str) -> Structure:
    """
    Read in PDB file into Biopython Structure
    """
    parser = PDBParser(QUIET=True)
    return parser.get_structure('', pdbfile)


def structure2seq(pdbmodel: Structure) -> str:
    """
    Get the sequence of a PDB structure (takes sequence of first chain)
    
    adapted from bioinformatics.stackexchange.com/questions/14101
    """

    d3to1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 
             'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 
             'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

    # iterate each model, chain, and residue
    for model in pdbmodel:
        for chain in model:
            seq = []
            for residue in chain:
                # ignore special residues & hetatoms
                if residue.resname in d3to1:
                    seq.append(d3to1[residue.resname])
            return "".join(seq)


def sequenceAlignment(seq1: str, seq2: str, substitution_matrix: str="BLOSUM62") -> Align.PairwiseAlignments:
    """
    Create global pairwise alignment of two sequences

    alternative substitution matrices:
    'BENNER22', 'BENNER6', 'BENNER74', 'BLASTN', 'BLASTP', 'BLOSUM45', 'BLOSUM50', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 
    'DAYHOFF', 'FENG', 'GENETIC', 'GONNET1992', 'HOXD70', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78', 
    'MEGABLAST', 'NUC.4.4', 'PAM250', 'PAM30', 'PAM70', 'RAO', 'RISLER', 'SCHNEIDER', 'STR', 'TRANS'
    """
    matrix = Align.substitution_matrices.load(substitution_matrix)
    aligner = Align.PairwiseAligner(substitution_matrix=matrix)
    alignments = aligner.align(seq1, seq2)
    return alignments


def getCoordinates(strAlignment: StructureAlignment) -> tuple[list[float], list[float]]:
    """
    Get coordinates of aligned atoms for reference & sequence
    """
    coordinates = [(referenceResidues["CA"].coord, sampleResidues["CA"].coord) 
               for referenceResidues, sampleResidues in strAlignment.get_iterator() 
               if referenceResidues and sampleResidues]

    return zip(*coordinates)


def superImposition(strAlignment: StructureAlignment) -> tuple[np.ndarray, np.ndarray, np.float64]:
    """
    Impose structures and get rotation, translation and rmsd
    """
    referenceCoord, sampleCoord = getCoordinates(strAlignment)
    
    sup = SVDSuperimposer()
    sup.set(np.array(referenceCoord), np.array(sampleCoord))
    sup.run()
    rot, tran = sup.get_rotran()
    rmsd = sup.get_rms()
    return rot, tran, rmsd
    

def findBestAlignment(strAlignments: list[StructureAlignment]) -> StructureAlignment:
    """
    Return best alignment based on rmsd out of a list of structure alignments
    """
    min_rmsd = np.inf
    for strAlignment in strAlignments:
        _, _, rmsd = superImposition(strAlignment)
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            bestAlignment = strAlignment
    return bestAlignment


def calculateDeltas(bestAlignment: StructureAlignment) -> np.array:
    """
    Get distance between coordinate pairs
    """
    referenceCoord, sampleCoord = getCoordinates(bestAlignment)
    rot, tran, _ = superImposition(bestAlignment) 
    transformedSampleCoord = np.dot(sampleCoord, rot) + tran
    
    return np.linalg.norm(referenceCoord - transformedSampleCoord, axis=1)


def calculateRMSD(arr: list[float]) -> float:
    """   
    Calculate root means square deviation
    https://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions

    deprecated function - similar to SVDSuperimposer - sup.get_rms()
    """
    return np.sqrt(np.sum(np.linalg.norm(arr))**2/len(arr))


def structureAlignment(referencefile: str, samplefile: str, save_pdb: Union[str, bool] = False) -> tuple[np.float64, np.ndarray]:
    """
    create a structure alignment of structures with different lengths

    >>> structureAlignment("8ACV.pdb", "5IQT.pdb", "5IQT_imposed.pdb")
    """
    referenceModel = createModel(referencefile)
    sampleModel = createModel(samplefile)

    referenceSequence = structure2seq(referenceModel)
    sampleSequence = structure2seq(sampleModel)

    seqAlignments = sequenceAlignment(referenceSequence, sampleSequence)    

    if len(seqAlignments) > 100:
        warnings.warn(f"High number of sequence alignments: {len(seqAlignments)}")
    
    strAlignments = [StructureAlignment(seqAlignment, referenceModel, sampleModel) for seqAlignment in seqAlignments]
    bestAlignment = findBestAlignment(strAlignments)

    
    rot, tran, rmsd = superImposition(bestAlignment)
    deltas = calculateDeltas(bestAlignment)

    if save_pdb:
        sampleModel.transform(rot, tran)
        io=PDBIO()
        io.set_structure(sampleModel)
        io.save(str(save_pdb))
        print(f"Aligned structure saved as {save_pdb}")

    return rmsd, deltas

if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    directory = Path.cwd()      

    reference = "8acv.pdb"
    sample = "5iqt.pdb"

    referencefile = directory / reference
    samplefile = directory / sample
    imposedfile = directory / "5iqt_imposed.pdb"

    rmsd, deltas = structureAlignment(referencefile, samplefile, save_pdb=imposedfile)

    plt.plot(deltas)
    print("RMSD:", rmsd)
