# @Created Date: 2025-01-08 04:43:40 pm
# @Filename: io.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-01-11 12:29:45 pm
import math
import torch
import roma
from warnings import warn
from .utils import planar_angle, dihedral, projection


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


def featurize_frames(fa_v: torch.Tensor, fa_p: torch.Tensor, fb_v: torch.Tensor, fb_p: torch.Tensor):  # cutoff: float = 16.0
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
    rab = fb_p.unsqueeze(-3) - fa_p.unsqueeze(-2) # shape: (...xaxbx3)
    rba = -rab
    
    dab = rab.norm(dim=-1) # shape: (...xaxb)
    # dmask = dab < cutoff
    
    fa_vx = fa_v[..., 0].unsqueeze(-2) # shape: (...xax1x3)
    fa_vz = fa_v[..., 2].unsqueeze(-2) # shape: (...xax1x3)
    fb_vx = fb_v[..., 0].unsqueeze(-3) # shape: (...x1xbx3)
    fb_vz = fb_v[..., 2].unsqueeze(-3) # shape: (...x1xbx3)

    ta = planar_angle(fa_vz, rab)
    tb = planar_angle(fb_vz, rba)

    pa = planar_angle(fa_vx, rab - projection(rab, fa_vz)) # project on the z axis and then substract it
    pb = planar_angle(fb_vx, rba - projection(rba, fb_vz)) # project on the z axis and then substract it

    chi = torch.pi + dihedral(-fa_vz, rab, fb_vz)
    
    return dab, ta, tb, pa, pb, chi


def discretize_features(br: torch.Tensor, theta: torch.Tensor, dpsi: torch.Tensor, dchi: float, nring: int, ncellsring: torch.Tensor,
                        dab: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, chi: torch.Tensor):
    '''
    Discretize the 6D features into bin indices.
    
    NOTE: this function is not differentiable
    '''
    ir = torch.clip(torch.bucketize(dab, br, right=False) - 1, min=0)
    ita = torch.clip(torch.bucketize(ta, theta, right=False), min=0, max=nring-1)
    itb = torch.clip(torch.bucketize(tb, theta, right=False), min=0, max=nring-1)
    ipa = (pa / dpsi[ita]).to(dtype=torch.int64)
    ipb = (pb / dpsi[itb]).to(dtype=torch.int64)
    ic = (chi / dchi).to(dtype=torch.int64)
    
    nita = ncellsring[ita]
    ipa_mask = ipa >= nita
    if ipa_mask.any():
        warn('ipa_mask'); ipa[ipa_mask] = nita[ipa_mask] - 1
    
    nitb = ncellsring[itb]
    ipb_mask = ipb >= nitb
    if ipb_mask.any():
        warn('ipb_mask'); ipb[ipb_mask] = nitb[ipb_mask] - 1
    
    ic_thr = round(2 * math.pi / dchi)
    ic_mask = ic >= ic_thr
    if ic_mask.any():
        warn('ic_mask'); ic[ic_mask] = ic_thr - 1

    return ir, ita, itb, ipa, ipb, ic


def korp_energy(
        dab: torch.Tensor, ta: torch.Tensor, tb: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, chi: torch.Tensor,
        seqab: torch.Tensor, seqsepab: torch.Tensor,
        korp_maps_flatten: torch.Tensor, korp_maps_shape: torch.Size, fmapping: torch.Tensor, smapping: torch.Tensor,
        br: torch.Tensor, theta: torch.Tensor, dpsi: torch.Tensor, dchi: float, nring: int, ncellsring: torch.Tensor, icell: torch.Tensor,
        bonding_thr: int = 9):
    '''
    Input shape:
        seqab: (...xaxbx2)
        seqsepab: (...xaxb)

    Output shape: (...xaxb)
    
    NOTE: this function is not differentiable

    Additional NOTE:
    
    * Triu part
    * dab < dab_max
    * bond = 0 if (seq_sep > bonding_thr) or (diff_chain) else 1
    * bonding_factor = 1.8
    
    '''
    ir, ita, itb, ipa, ipb, ic = discretize_features(br, theta, dpsi, dchi, nring, ncellsring, dab, ta, tb, pa, pb, chi)
    
    dab_min, dab_max = br[0], br[-1]
    mask = ((seqsepab > 0) & (dab > dab_min) & (dab < dab_max)).to(dtype=torch.int64) # shape (...xaxb)
    
    sd = seqsepab.clone()     # | -> when `smapping=[ 0, -1,  1,  1,  1,  0,  0,  0,  0,  0]`, `bonding_thr` is essentially 4 and `seqsepab` should > 1
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

