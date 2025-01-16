# PyKORP

Python bindings and PyTorch implementation of [KORP](https://github.com/chaconlab/Korp).

## Installation

```bash
pip install pykorp # optional: pip install pykorp[ckorp]
```

## Usage

```python
import pykorp
from pykorp import frame_coords, featurize_frames, korp_energy
import torch

device = 'cpu' # or 'cuda:0'

try:
    config = pykorp.config('korp6Dv1.bin', device=device)
except Exception:
    config = torch.load('korp6Dv1.bin.pt', map_location=device)

chain_info, n_coords, ca_coords, c_coords, seqab, seqsepab = pykorp.pdb_io('2DWV.cif.gz', device=device)

features = featurize_frames(frame_coords(n_coords, ca_coords, c_coords), ca_coords, mask=seqsepab > 1)

korpe = korp_energy(features, seqab, seqsepab, config)
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

If you use this software in your work, please also cite:

```bibtex
@software{PyKORP_2025,
  author  = {Zhu, Zefeng},
  license = {MIT},
  title   = {{PyKORP: Python bindings and PyTorch implementation of KORP.}},
  url     = {https://github.com/naturegeorge/pykorp},
  version = {1.0.0},
  year    = {2025},
  month   = {01}
}
```