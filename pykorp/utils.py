# Copyright 2025 Zefeng Zhu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @Created Date: 2025-01-08 08:35:40 pm
# @Filename: utils.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2025-01-09 11:51:08 am
import torch


def planar_angle(b1: torch.Tensor, b2: torch.Tensor):
    #b1 = point0 - point1
    #b2 = point2 - point1
    return torch.atan2(
        torch.linalg.cross(b1, b2).norm(dim=-1),
        torch.einsum('...km,...km->...k', b1, b2))


def dihedral(vec01: torch.Tensor, vec12: torch.Tensor, vec23: torch.Tensor):
    #b = point2 - point1
    #u = torch.linalg.cross(b, point1 - point0)
    #w = torch.linalg.cross(b, point2 - point3)
    b = vec12
    u = torch.linalg.cross(b, vec01)
    w = torch.linalg.cross(b, -vec23)
    #if len(b.shape) > 1:
    return torch.atan2(
            torch.einsum('...km,...km->...k', torch.linalg.cross(u, w), b),
            torch.mul(torch.einsum('...km,...km->...k', u, w), b.norm(dim=-1)))
    #else:
    #    return torch.atan2(torch.linalg.cross(u, w).dot(b), u.dot(w) * torch.linalg.norm(b))


def projection(vec: torch.Tensor, ref: torch.Tensor):
    aTb = torch.einsum('...km,...km->...k', ref, vec)
    aTa = torch.einsum('...km,...km->...k', ref, ref)
    return (aTb/aTa).unsqueeze(-1) * ref
    
