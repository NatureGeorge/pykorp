# PyKORP

Python bindings and PyTorch implementation of [KORP](https://github.com/chaconlab/Korp).

## Installation

```bash
pip install pykorp # optional: pip install pykorp[ckorp]
```

## Usage

```python
import pykorp
from pykorp.io import frame_coords, featurize_frames, korp_energy
import torch

device = 'cpu' # or 'cuda:0'

try:
    config = pykorp.config('korp6Dv1.bin', device=device)
except Exception:
    config = torch.load('korp6Dv1.bin.pt', map_location=device)

chain_info, n_coords, ca_coords, c_coords, seqab, seqsepab = pykorp.pdb_io('2DWV.cif.gz', device=device)

korpe = korp_energy(
        *featurize_frames(frame_coords(n_coords, ca_coords, c_coords), ca_coords),
        seqab, seqsepab,
        *config).sum(dim=(-1, -2))
```

## Reference

```bibtex
@article{10.1093/bioinformatics/btz026,
  author   = {López-Blanco, José Ramón and Chacón, Pablo},
  title    = {{KORP: knowledge-based 6D potential for fast protein and loop modeling}},
  journal  = {Bioinformatics},
  volume   = {35},
  number   = {17},
  pages    = {3013-3019},
  year     = {2019},
  month    = {01},
  issn     = {1367-4803},
  doi      = {10.1093/bioinformatics/btz026}
}
```