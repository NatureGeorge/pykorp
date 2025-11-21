# @Created Date: 2025-01-08 04:43:40 pm
# @Filename: feat.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-11-21 03:48:52 pm
import math
import itertools
import operator
import torch
import roma
import numpy as np
from warnings import warn
from typing import Optional, Tuple, Dict
from .utils import planar_angle, dihedral, projection


aa_dict = dict(
    ALA =  0,
    CYS =  1,
    ASP =  2,
    GLU =  3,
    PHE =  4,
    GLY =  5,
    HIS =  6,
    ILE =  7,
    LYS =  8,
    LEU =  9,
    MET =  10,
    ASN =  11,
    PRO =  12,
    GLN =  13,
    ARG =  14,
    SER =  15,
    THR =  16,
    VAL =  17,
    TRP =  18,
    TYR =  19,
    SEC = 1,
    PYL = 8,
)


aa20_one_letter_code = (
 'A',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'V',
 'W',
 'Y')


aa20_three_letter_code = (
    'ALA',
    'CYS',
    'ASP',
    'GLU',
    'PHE',
    'GLY',
    'HIS',
    'ILE',
    'LYS',
    'LEU',
    'MET',
    'ASN',
    'PRO',
    'GLN',
    'ARG',
    'SER',
    'THR',
    'VAL',
    'TRP',
    'TYR',
)


W6DK = torch.tensor([1.875, 0.718, 0.876, 0.876, 2.440, 0.973, 1.030, 2.512, 1.030, 2.512, 1.643, 1.113, 1.411, 1.113, 1.608, 1.113, 1.113, 2.512, 1.411, 2.440])


def frame_coords(n_coords: torch.Tensor, ca_coords: torch.Tensor, c_coords: torch.Tensor):
    '''
    Definition of the NCaC frame used by 
    
    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.

    Input shape: (...x3)
    Output shape: (...x3x3)

    ```python
    # Raw
    r_cca = c_coords - ca_coords
    r_nca = n_coords - ca_coords

    v_z = torch.nn.functional.normalize(r_cca + r_nca, dim=-1)
    v_y = torch.nn.functional.normalize(torch.cross(v_z, r_nca))
    v_x = torch.cross(v_y, v_z)

    rotmat = torch.stack((v_x, v_y, v_z), dim=2)
    ```
    '''
    # return roma.special_gramschmidt(torch.stack([c_coords - ca_coords, n_coords - ca_coords], dim=2)) # typical NCaC frame
    r_cca = c_coords - ca_coords
    r_nca = n_coords - ca_coords
    return roma.special_gramschmidt(torch.stack([r_cca + r_nca, r_nca], dim=-1)).roll(shifts=-1, dims=-1) # the NCaC frame used by KORP


def featurize_frames_full(fa_v: torch.Tensor, fa_p: torch.Tensor, fb_v: Optional[torch.Tensor] = None, fb_p: Optional[torch.Tensor] = None):  # cutoff: float = 16.0
    '''
    Definition of the 6D features (relative orientation and position) used by 
    
    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.
    
    Input shape:
        fa_v: rotmat_a (...xax3x3)
        fa_p: trans_a (...xax3)
        fb_v: rotmat_b (...xbx3x3)
        fb_p: trans_b (...xbx3)
    
    Output shape: (...xaxbx6)
    '''
    if fb_v is None: fb_v = fa_v
    if fb_p is None: fb_p = fa_p
    is_symmetric = (fa_v is fb_v) and (fa_p is fb_p)

    rab = fb_p.unsqueeze(-3) - fa_p.unsqueeze(-2) # shape: (...xaxbx3)
    rba = -rab
    
    dab = rab.norm(dim=-1) # shape: (...xaxb)
    # dmask = dab < cutoff
    
    fa_vx = fa_v[..., 0].unsqueeze(-2) # shape: (...xax1x3)
    fa_vy = fa_v[..., 1].unsqueeze(-2) # shape: (...xax1x3)
    fa_vz = fa_v[..., 2].unsqueeze(-2) # shape: (...xax1x3)

    ta = planar_angle(fa_vz, rab)

    rab_on_axy = rab - projection(rab, fa_vz) # project on the z axis and then substract it
    pa = torch.pi - planar_angle(fa_vx, rab_on_axy)
    keep_maska = torch.einsum('...km,...km->...k', fa_vy, rab_on_axy) < 0
    pa[~keep_maska] *= -1
    pa += torch.pi

    if is_symmetric:
        tb = None # ta.transpose(-1, -2)
        pb = None # pa.transpose(-1, -2)
        fb_vz = fb_v[..., 2].unsqueeze(-3) # shape: (...x1xbx3)
    else:
        fb_vx = fb_v[..., 0].unsqueeze(-3) # shape: (...x1xbx3)
        fb_vy = fb_v[..., 1].unsqueeze(-3) # shape: (...x1xbx3)
        fb_vz = fb_v[..., 2].unsqueeze(-3) # shape: (...x1xbx3)

        tb = planar_angle(fb_vz, rba)

        rba_on_bxy = rba - projection(rba, fb_vz) # project on the z axis and then substract it
        pb = torch.pi - planar_angle(fb_vx, rba_on_bxy)
        keep_maskb = torch.einsum('...km,...km->...k', fb_vy, rba_on_bxy) < 0
        pb[~keep_maskb] *= -1
        pb += torch.pi

    chi = torch.pi + dihedral(fa_vz, rba, -fb_vz)
    
    return dab, ta, tb, pa, pb, chi


def featurize_frames(fa_v: torch.Tensor, fa_p: torch.Tensor, fb_v: Optional[torch.Tensor] = None, fb_p: Optional[torch.Tensor] = None, dab_min: float = 3, dab_max: float = 16, mask : Optional[torch.Tensor] = None, return_joint_concat: bool = False):
    '''
    Definition of the 6D features (relative orientation and position) used by 
    
    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.
    
    Input shape:
        fa_v: rotmat_a (...xax3x3)
        fa_p: trans_a (...xax3)
        fb_v: rotmat_b (...xbx3x3)
        fb_p: trans_b (...xbx3)
        mask: (...xaxb)
    
    TODO: apply sth like `pytorch3d.ops.ball_query`
    '''
    if fb_v is None: fb_v = fa_v
    if fb_p is None: fb_p = fa_p
    is_symmetric = (fa_v is fb_v) and (fa_p is fb_p)

    rab = fb_p.unsqueeze(-3) - fa_p.unsqueeze(-2) # shape: (...xaxbx3) # NOTE: out of memory keypoint
    
    dab = rab.norm(dim=-1) # shape: (...xaxb)
    dmask = (dab < dab_max) & (dab > dab_min) # shape: (...xaxb)
    if return_joint_concat: joint_contact = dmask.any(dim=0)
    if mask is not None: dmask &= mask
    index_ab = torch.where(dmask)
    
    dab = dab[dmask]
    rab = rab[dmask]
    rba = -rab
    
    fa_v = fa_v[index_ab[:-1]]
    fb_v = fb_v[index_ab[:-2]+(index_ab[-1],)]
    fa_vx = fa_v[..., 0]
    fa_vy = fa_v[..., 1]
    fa_vz = fa_v[..., 2]

    ta = planar_angle(fa_vz, rab)

    rab_on_axy = rab - projection(rab, fa_vz) # project on the z axis and then substract it
    pa = torch.pi - planar_angle(fa_vx, rab_on_axy)
    keep_maska = torch.einsum('...km,...km->...k', fa_vy, rab_on_axy) < 0
    pa[~keep_maska] *= -1
    pa += torch.pi

    
    fb_vx = fb_v[..., 0]
    fb_vy = fb_v[..., 1]
    fb_vz = fb_v[..., 2]

    tb = planar_angle(fb_vz, rba)

    rba_on_bxy = rba - projection(rba, fb_vz) # project on the z axis and then substract it
    pb = torch.pi - planar_angle(fb_vx, rba_on_bxy)
    keep_maskb = torch.einsum('...km,...km->...k', fb_vy, rba_on_bxy) < 0
    pb[~keep_maskb] *= -1
    pb += torch.pi

    chi = torch.pi + dihedral(fa_vz, rba, -fb_vz)
    
    if return_joint_concat: return ((dab, ta, tb, pa, pb, chi), index_ab), joint_contact
    return (dab, ta, tb, pa, pb, chi), index_ab


def discretize_features(br: torch.Tensor, theta: torch.Tensor, dpsi: torch.Tensor, dchi: float, nring: int, ncellsring: torch.Tensor,
                        dab: torch.Tensor, ta: torch.Tensor, tb: Optional[torch.Tensor], pa: torch.Tensor, pb: Optional[torch.Tensor], chi: torch.Tensor):
    '''
    Discretize the 6D features into bin indices.
    
    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.
    
    NOTE: this function is not differentiable
    '''
    ir = torch.clip(torch.bucketize(dab, br, right=False) - 1, min=0)
    ita = torch.clip(torch.bucketize(ta, theta, right=False), min=0, max=nring-1)
    ipa = (pa / dpsi[ita]).to(dtype=torch.int64)
    ic = (chi / dchi).to(dtype=torch.int64)
    
    nita = ncellsring[ita]
    ipa_mask = ipa >= nita
    if ipa_mask.any():
        warn(f'ipa_mask: {ipa_mask.sum()}'); ipa[ipa_mask] = nita[ipa_mask] - 1
    
    if tb is None:
        assert pb is None
        itb = None#ita.transpose(-1, -2)
        ipb = None#ipa.transpose(-1, -2)
    else:
        itb = torch.clip(torch.bucketize(tb, theta, right=False), min=0, max=nring-1)
        ipb = (pb / dpsi[itb]).to(dtype=torch.int64)

        nitb = ncellsring[itb]
        ipb_mask = ipb >= nitb
        if ipb_mask.any():
            warn(f'ipb_mask: {ipb_mask.sum()}'); ipb[ipb_mask] = nitb[ipb_mask] - 1

    ic_thr = round(2 * math.pi / dchi)
    ic_mask = ic >= ic_thr
    if ic_mask.any():
        warn(f'ic_mask: {ic_mask.sum()}'); ic[ic_mask] = ic_thr - 1

    return ir, ita, itb, ipa, ipb, ic


def korp_energy_full(
        dab: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, chi: torch.Tensor,
        seqab: torch.Tensor, seqsepab: torch.Tensor,
        korp_maps_flatten: torch.Tensor, korp_maps_shape: torch.Size, fmapping: torch.Tensor, smapping: torch.Tensor,
        br: torch.Tensor, theta: torch.Tensor, dpsi: torch.Tensor, dchi: float, nring: int, ncellsring: torch.Tensor, icell: torch.Tensor,
        bonding_thr: int = 9):
    '''
    Calculate the KORP energy introduced by

    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.

    Input shape:
        seqab: (...xaxbx2)
        seqsepab: (...xaxb)

    Output shape: (...xaxb)
    
    NOTE: this function is not differentiable

    Additional NOTE:
    
    * Triu part
    * dab < dab_max
    * bond = 0 if (seq_sep > bonding_thr) or (diff_chain) else 1
    
    Typical usage:

    ```python
    device = 'cpu'
    bonding_thr = 9
    bonding_factor = 1.8
    config = pykorp.config('korp6Dv1.bin', device=device, bonding_factor=bonding_factor)
    chain_info, n_coords, ca_coords, c_coords, seqab, seqsepab = pykorp.pdb_io('2KOX.cif.gz', device=device)

    korpe = korp_energy_full(
            *featurize_frames_full(frame_coords(n_coords, ca_coords, c_coords), ca_coords),
            seqab, seqsepab,
            korp_map.flatten(), korp_map.shape, *config[1:],
            bonding_thr=bonding_thr).sum(dim=(-1, -2))
    ```

    '''
    ir, ita, itb, ipa, ipb, ic = discretize_features(br, theta, dpsi, dchi, nring, ncellsring, dab, ta, tb, pa, pb, chi)

    if itb is None:
        itb = ita.transpose(-1, -2)
    if ipb is None:
        ipb = ipa.transpose(-1, -2)
    
    dab_min, dab_max = br[0], br[-1]
    mask = ((seqsepab > 0) & (dab > dab_min) & (dab < dab_max)).to(dtype=torch.int64) # shape (...xaxb)
    
    sd = seqsepab.abs()       # | -> when `smapping=[ 0, -1,  1,  1,  1,  0,  0,  0,  0,  0]`, `bonding_thr` is essentially 4 and `seqsepab` should > 1
    sd[sd > bonding_thr] = 0  # | -> thus `mask` should be `((seqsepab > 1) & (dab > dab_min) & (dab < dab_max)).to(dtype=torch.int64)`
    s = smapping[sd]          # | -> then `s` can be directly assigned to be `(seqsepab <= bonding_thr).to(dtype=torch.int64)` given that `bonding_thr = 4` # shape (...xaxb)
    mask &= s >= 0            # | -> but here remain the original code for potential flexibility
    
    weighting_factor = fmapping[s] # shape (...xaxb)
    # korp_maps,                   # original len(shape) eq 7 -> assume already be flatten
    korp_maps_strides = [math.prod(korp_maps_shape[i+1:]) for i in range(len(korp_maps_shape))]
    #map_index = (s,                # shape (...xaxb) 
    #             seqab,            # shape (...xaxbx2)
    #             ir,               # shape (...xaxb) 
    #             icell[ita] + ipa, # shape (...xaxb) 
    #             icell[itb] + ipb, # shape (...xaxb) 
    #             ic                # shape (...xaxb) 
    #            )
    map_index = (s *                 korp_maps_strides[0] +
                seqab[..., 0] *      korp_maps_strides[1] +
                seqab[..., 1] *      korp_maps_strides[2] +
                ir *                 korp_maps_strides[3] +
                (icell[ita] + ipa) * korp_maps_strides[4] +
                (icell[itb] + ipb) * korp_maps_strides[5] +
                ic *                 korp_maps_strides[6])
    energy = torch.take(korp_maps_flatten, map_index) # shape (...xaxb) 
    energy = mask * weighting_factor * energy
    
    return energy #.sum(dim=(-1,-2))


def korp_energy_raw(
        dab: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, chi: torch.Tensor,
        seqab: torch.Tensor, seqsepab: torch.Tensor,
        korp_maps: torch.Tensor, fmapping: torch.Tensor, smapping: torch.Tensor,
        br: torch.Tensor, theta: torch.Tensor, dpsi: torch.Tensor, dchi: float, nring: int, ncellsring: torch.Tensor, icell: torch.Tensor,
        bonding_thr: int = 4):
    '''
    Calculate the KORP energy introduced by

    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.

    Input shape:
        seqab: (...xsx2)
        seqsepab: (...xs)

    Output shape: (...xs)
    
    NOTE: this function is not differentiable
    '''
    ir, ita, itb, ipa, ipb, ic = discretize_features(br, theta, dpsi, dchi, nring, ncellsring, dab, ta, tb, pa, pb, chi)
    s = (seqsepab <= bonding_thr).to(torch.int64) # when `smapping=[ 0, -1,  1,  1,  1,  0,  0,  0,  0,  0]`, `bonding_thr` is essentially 4 
    energy = fmapping[s] * korp_maps[s, seqab[..., 0], seqab[..., 1], ir, (icell[ita] + ipa), (icell[itb] + ipb), ic]
    return energy


def energy_gather(batch_shape, energy: torch.Tensor, index, end_dim: int):
    flat_index = sum(i*j for i,j in zip(tuple(itertools.accumulate(batch_shape[::-1], operator.mul, initial=1))[::-1][-len(batch_shape):], index[:end_dim]))
    result = torch.zeros(math.prod(batch_shape), dtype=energy.dtype, device=energy.device).scatter_add_(dim=0, index=flat_index, src=energy).view(batch_shape)
    return result


def korp_energy(features: Tuple, seqab: torch.Tensor, seqsepab: torch.Tensor, config: Tuple, per_residue: bool = False, per_residue_sym: bool = False):
    '''
    Calculate the KORP energy introduced by

    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.

    NOTE: more flexible settings can be achieved via `korp_energy_raw`.

    Input shape:
        seqab: (...xaxbx2)
        seqsepab: (...xaxb)

    Output shape: (...) which is determined by `features`
    
    NOTE: this function is not differentiable

    Typical usage:

    ```python
    device = 'cuda:0'
    config = pykorp.config('korp6Dv1.bin', device=device)
    chain_info, n_coords, ca_coords, c_coords = pykorp.pdb_io('2DWV.cif.gz', device=device)
    modnum = n_coords.shape[0]
    seqab, seqsepab = pykorp.seq_info(chain_info['seq'], chain_info['seq_index'], chain_info['length'])
    features = featurize_frames(frame_coords(n_coords, ca_coords, c_coords), ca_coords, mask=seqsepab > 1)
    korpe = korp_energy(features, seqab.expand(modnum, *seqab.shape[1:]), seqsepab.expand(modnum, *seqsepab.shape[1:]), config)
    '''
    features, index = features
    energy = korp_energy_raw(
            *features,
            seqab[index], seqsepab[index],
            *config)
    
    if not per_residue:
        end_dim = -2
    else:
        index = list(index)
        a = index[-2]
        b = index[-1]
        index[-2] = b
        index[-1] = a
        end_dim = -1
    
    batch_shape = tuple(i.max().cpu().tolist()+1 for i in index[:end_dim])
    result = energy_gather(batch_shape, energy, index, end_dim)
    
    if per_residue and per_residue_sym:
        a = index[-2]
        b = index[-1]
        index[-2] = b
        index[-1] = a
        result += energy_gather(batch_shape, energy, index, end_dim)
    
    energy = result
    
    return energy


def korpm_energy(features: Tuple, seqab: torch.Tensor, seqsepab: torch.Tensor, config: Tuple, mutations = None, return_cache: bool = False, diff_index_dict: Optional[Dict] = None, cache_dict: Optional[Dict] = None, cal_cache_dict: bool = False, remaining_cache: Optional[Tuple] = None):
    '''
    Calculate ddG using the KORP energy introduced by

    Iván Martín Hernández, Yves Dehouck, Ugo Bastolla, José Ramón López-Blanco, Pablo Chacón. (2023)
    Predicting protein stability changes upon mutation using a simple orientational potential.
    doi: 10.1093/bioinformatics/btad011.

    José Ramón López-Blanco and Pablo Chacón. (2019)
    KORP: knowledge-based 6D potential for fast protein and loop modeling.
    doi: 10.1093/bioinformatics/btz026.

    Input shape:
        seqab: (...xaxbx2)
        seqsepab: (...xaxb)
        mutations: format: Sequence((position_idx, muta_type_idx)); None for single-point deep mutational scanning

    Output shape: (...) which is determined by `features`
    
    NOTE: this function is not differentiable

    Typical usage:

    ```python
    device = 'cuda:0'
    config = pykorp.config('korp6Dv1.bin', device=device)
    chain_info, n_coords, ca_coords, c_coords = pykorp.pdb_io('2LHD.cif.gz', device=device)
    modnum = n_coords.shape[0]
    seqab, seqsepab = pykorp.seq_info(chain_info['seq'], chain_info['seq_index'], chain_info['length'])
    features = featurize_frames(frame_coords(n_coords, ca_coords, c_coords), ca_coords, mask=seqsepab > 1)
    ddG = korpm_energy(features, seqab.expand(modnum, *seqab.shape[1:]), seqsepab.expand(modnum, *seqsepab.shape[1:]), config, mutations=[(24, 7), (44, 9)])
    '''
    W6D = W6DK.to(device=config[0].device)
    batch_size = seqab.shape[0]
    length = seqab.shape[1]

    features_, index = features
    if remaining_cache is not None:
        s, fmapping_s, seqab_index, ir, icell_ita_ipa, icell_itb_ipb, ic = remaining_cache
    else:
        seqab_index = seqab[index]
        seqsepab_index = seqsepab[index]

        dab, ta, tb, pa, pb, chi = features_
        fmapping, smapping, br, theta, dpsi, dchi, nring, ncellsring, icell = config[1:]
        ir, ita, itb, ipa, ipb, ic = discretize_features(br, theta, dpsi, dchi, nring, ncellsring, dab, ta, tb, pa, pb, chi)
        s = (seqsepab_index <= 4).to(torch.int64)
        icell_ita_ipa, icell_itb_ipb = (icell[ita] + ipa), (icell[itb] + ipb)
        fmapping_s = fmapping[s]
    
    if mutations is None:
        DMS = True
        positions = range(length)
        mutations = [(loc, aa_idx) for loc in positions for aa_idx in range(20)]
    elif isinstance(mutations, int):
        DMS = False
        loc = mutations
        positions = [loc]
        mutations = [(loc, aa_idx) for aa_idx in range(20)]
    elif isinstance(mutations, (list, torch.Tensor, np.ndarray)):
        DMS = True
        positions = list(mutations)
        mutations = [(loc, aa_idx) for loc in positions for aa_idx in range(20)]
    else:
        DMS = False
        positions = frozenset(muta[0] for muta in mutations)
    
    if diff_index_dict is None:
        diff_index_dict = dict()
        cal_diff_index = True
    else:
        cal_diff_index = False
    if cache_dict is None:
        cache_dict = dict()
        cal_cache_dict = True
    
    if cal_diff_index or cal_cache_dict:
        for loc in positions:
            if cal_diff_index:
                muta_seqab = seqab.clone()
                muta_seqab[:, loc, :, 0] = muta_seqab[:, :, loc, 1] = -1
                muta_seqab_index = muta_seqab[index]
                diff_index = torch.where(seqab_index != muta_seqab_index)[0]
                diff_index_dict[loc] = diff_index
            else:
                diff_index = diff_index_dict[loc]

            if cal_cache_dict:
                fmapping_s_diff_indexed = fmapping_s[diff_index]
                s_diff_indexed = s[diff_index]
                seqab_index_diff_indexed = seqab_index[diff_index]
                ir_diff_indexed = ir[diff_index]
                icell_ita_ipa_diff_indexed = icell_ita_ipa[diff_index]
                icell_itb_ipb_diff_indexed = icell_itb_ipb[diff_index]
                ic_diff_indexed = ic[diff_index]
                wild_type_dG = fmapping_s_diff_indexed * config[0][s_diff_indexed, seqab_index_diff_indexed[...,0], seqab_index_diff_indexed[...,1], ir_diff_indexed, icell_ita_ipa_diff_indexed, icell_itb_ipb_diff_indexed, ic_diff_indexed] * W6D[seqab_index_diff_indexed.flatten()].reshape(seqab_index_diff_indexed.shape).prod(dim=-1)
                muta_type_dG_cache = fmapping_s_diff_indexed.unsqueeze(-1).unsqueeze(-1) * config[0][s_diff_indexed, :, :, ir_diff_indexed, icell_ita_ipa_diff_indexed, icell_itb_ipb_diff_indexed, ic_diff_indexed]

                cache_dict[loc] = (wild_type_dG, muta_type_dG_cache)

    ddG = []
    for pos, muta_aa_idx in mutations:
        muta_seqab = seqab.clone()
        muta_seqab[:, pos, :, 0] = muta_seqab[:, :, pos, 1] = muta_aa_idx
        muta_seqab_index = muta_seqab[index]
        muta_seqab_index_diff_indexed = muta_seqab_index[diff_index_dict[pos]]

        wild_type_dG, muta_type_dG_cache = cache_dict[pos]
        # TODO: broadcast `wild_type_dG`, stack `muta_type_dG` ?
        x = ((wild_type_dG - muta_type_dG_cache[torch.arange(muta_type_dG_cache.shape[0]), muta_seqab_index_diff_indexed[..., 0], muta_seqab_index_diff_indexed[..., 1]] * W6D[muta_seqab_index_diff_indexed.flatten()].reshape(muta_seqab_index_diff_indexed.shape).prod(dim=-1))/100)

        end_dim = -2
        diff_index_tuple = tuple(i[diff_index_dict[pos]] for i in index)
        batch_shape = (batch_size, ) # tuple(i.max().cpu().tolist()+1 for i in diff_index_tuple[:end_dim])
        ddG.append(energy_gather(batch_shape, x, diff_index_tuple, end_dim))
    
    if ddG: ddG = torch.stack(ddG, dim=0)
    if DMS: ddG = ddG.reshape(-1, 20, batch_size)
    if return_cache: return ddG, diff_index_dict, cache_dict, [s, fmapping_s, seqab_index, ir, icell_ita_ipa, icell_itb_ipb, ic]
    return ddG
