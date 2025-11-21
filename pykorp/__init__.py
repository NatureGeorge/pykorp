# @Created Date: 2025-01-07 02:40:14 pm
# @Filename: __init__.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-11-21 09:46:03 pm
import torch
import gemmi
from collections import defaultdict
from .feat import aa_dict as AA_20
from .feat import W6DK as W6D
from .feat import frame_coords, featurize_frames, featurize_frames_full, discretize_features, korp_energy_full, korp_energy_raw, korp_energy
from .feat import korpm_energy, korpm_evo_unit, dfs_to_trajectories

def config(bin_path: str, device: str = 'cpu', bonding_factor: float = 1.8):
    from .c_korp import read_korp
    
    data = read_korp(bin_path, bonding_factor)
    maps = data['maps'][:,0,0]
    korp_maps = torch.from_numpy(maps).to(device=device)
    fmapping = torch.from_numpy(data['fmapping']).to(device=device)
    smapping = torch.from_numpy(data['smapping']).to(dtype=torch.long, device=device)
    br = torch.from_numpy(data['br']).to(device=device)
    theta = torch.from_numpy(data['meshes'][0]['theta']).to(device=device)
    dpsi = torch.from_numpy(data['meshes'][0]['dpsi']).to(device=device)
    dchi = data['meshes'][0]['dchi']
    nring = data['meshes'][0]['nring']
    ncellsring = torch.from_numpy(data['meshes'][0]['ncellsring']).to(dtype=torch.long, device=device)
    icell = torch.from_numpy(data['meshes'][0]['icell']).to(dtype=torch.long, device=device)
    return korp_maps, fmapping, smapping, br, theta, dpsi, dchi, nring, ncellsring, icell


def seqid2num(seqid, record):
    record[seqid.num] += 1
    gap = sum(val - 1 for key, val in record.items() if key < seqid.num and val > 1)
    return seqid.num + record[seqid.num] - 1 + gap


def pdb_io(pdb_path: str, asym_ids = None, chain_ids = None, aa_dict = AA_20, device: str = 'cpu'):
    st = gemmi.read_structure(pdb_path)
    st.remove_alternative_conformations()

    if pdb_path.endswith('.cif') or pdb_path.endswith('.cif.gz'):
        if (asym_ids is None) and (chain_ids is None):
            asym_ids = []
            for entity in st.entities:
                if entity.polymer_type == gemmi.PolymerType.PeptideL:
                    asym_ids.extend(entity.subchains)
        elif (asym_ids is None) and (chain_ids is not None):
            asym_ids = []
    elif pdb_path.endswith('.pdb') or pdb_path.endswith('.pdb.gz'):
        asym_ids = []
    else:
        raise NotImplementedError('Invalid file suffix!')
    
    seq = []
    seq_index = []
    coords = []
    entity_ids = []
    for asym_id in asym_ids:
        for entity in st.entities:
            if asym_id in entity.subchains:
                entity_ids.append(entity.name)
        seq.append(torch.tensor([aa_dict[res.name] for res in st[0].get_subchain(asym_id)]))
        seq_index.append(torch.tensor([res.label_seq for res in st[0].get_subchain(asym_id)]))
        coords.append(torch.tensor([[[res['N'][0].pos.tolist(), res['CA'][0].pos.tolist(), res['C'][0].pos.tolist()] for res in mod.get_subchain(asym_id)] for mod in st]))
    
    if not asym_ids:
        if chain_ids is None:
            chain_ids = []
            for chain in st[0]:
                chain_id = chain.name
                chain = chain.get_polymer()
                if chain.check_polymer_type() == gemmi.PolymerType.PeptideL:
                    chain_ids.append(chain_id)
                    seq.append(torch.tensor([aa_dict[res.name] for res in chain]))
                    seqid_record = defaultdict(int)
                    seq_index.append(torch.tensor([seqid2num(res.seqid, seqid_record) for res in chain]))
                    coords.append(torch.tensor([[[res['N'][0].pos.tolist(), res['CA'][0].pos.tolist(), res['C'][0].pos.tolist()] for res in mod[chain_id].get_polymer()] for mod in st]))
        else:
            for chain_id in chain_ids:
                seq.append(torch.tensor([aa_dict[res.name] for res in st[0][chain_id].get_polymer()]))
                seqid_record = defaultdict(int)
                seq_index.append(torch.tensor([seqid2num(res.seqid, seqid_record) for res in st[0][chain_id].get_polymer()]))
                coords.append(torch.tensor([[[res['N'][0].pos.tolist(), res['CA'][0].pos.tolist(), res['C'][0].pos.tolist()] for res in mod[chain_id].get_polymer()] for mod in st]))
                    
        chain_info = {'chain_ids': chain_ids}
    else:
        chain_info = {'asym_ids': asym_ids}
        assert len(seq) > 0
    

    length = [len(i) for i in seq]
    seq = torch.cat(seq, dim=0).to(device=device)
    seq_index = torch.cat(seq_index, dim=0).to(device=device)
    coords = torch.cat(coords, dim=1).to(device=device)
    n_coords, ca_coords, c_coords = coords[:,:,0], coords[:,:,1], coords[:,:,2]
    
    chain_info['seq'] = seq
    chain_info['seq_index'] = seq_index
    chain_info['length'] = length
    chain_info['entity_id'] = entity_ids
    return chain_info, n_coords, ca_coords, c_coords

    
def seq_info(seq: torch.Tensor, seq_index: torch.Tensor, length):
    # TODO: optimize seqab and seqsepab -> make it memory efficient
    seqab = torch.stack(torch.meshgrid(seq, seq, indexing='ij'), dim=-1)[None]

    if len(length) > 1:
        seqsepab = torch.full((seq_index.shape[0], seq_index.shape[0]), fill_value=seq_index.shape[0], device=seq.device)
        seqsepab = torch.triu(seqsepab) - torch.tril(seqsepab)
        for chain_idx in range(len(length)):
            beg = sum(length[:chain_idx])
            end = beg + length[chain_idx]
            seqsepab[beg:end, beg:end] = seq_index[None, beg:end] - seq_index[beg:end, None]
        seqsepab = seqsepab.unsqueeze(0)
    else:
        seqsepab = (seq_index[None, :] - seq_index[:, None])[None]
    return seqab, seqsepab


def seq_info_serial(seq: torch.Tensor, seq_index: torch.Tensor, length):
    seqab = []
    seqsep_ab = []
    total_length = seq.shape[0]
    for i in range(total_length-1):
        for j in range(i+1, total_length):
            seqab.append(torch.stack((seq[i], seq[j])))
            seqsep_ab.append(seq_index[j] - seq_index[i] if True else total_length)
    return