# Matryoshka

A Python package for predicting the galaxy power spectrum with a neural network (NN)
based emulator.

> [!IMPORTANT]
> This is a modified, developmental version of Jamie Donald-McCann's Matryosha.
> There is no API change, except the underlying syntax has been updated to
> match dependency updates and the docstrings have been enhanced.

## Installation

The package can be installed by cloning this repository and using Pip in editable mode.

```
git clone https://github.com/JDonaldM/Matryoshka
cd Matryoshka
pip install -e .
```

## Basic usage

The example bellow shows how to generate a prediction for a Planck18 LCDM
transfer function using `matryoshka`.

```python
import matryoshka.emulator as matry
import numpy as np
from astropy.cosmology import Planck18

cosmo = np.array([
    Planck18.Om0,
    Planck18.Ob0,
    Planck18.h,
    Planck18.Neff,
    -1.0
])

transfer_emu = matry.Transfer()

emu_pred = transfer_emu.emu_predict(cosmo)
```

For more examples and full documentation, see
https://matryoshka-emu.readthedocs.io/en/latest/

## Change log

### v0.2.0

We include an emulator for predicicting multipoles of the power spectrum that would be
calculated using the EFT-of-LSS method. This EFT emulator provides a roughly 500 times
speed-up compared to the EFT-of-LSS code [PyBird](https://github.com/pierrexyz/pybird),
and prodcues predictions that are accurate within 1% (at 68% CI).

### v0.1.0

We include an emulator to predict the nonlinear boost for the *matter* power spectrum
that has been trained on the [Quijote simulations](https://arxiv.org/abs/1909.05273).
We also include a version of the transfer function emulator that has been trained on
the Quijote sample space.

## Nonlinear boost component emulator

> [!NOTE]
> In the current version of `matryoshka` the nonlinear boost component emulator has
> only been trained with training data generated with
> [HALOFIT](https://iopscience.iop.org/article/10.1088/0004-637X/761/2/152) and serves
> to demonstrate the use of `matryoshka`. Future versions will include a nonlinear boost
> component emulator trained with data produced with high resolution N-body simulatios.

## License & attribution

Copyright 2021 Jamie Donald-McCann. `matryoshka` is free to use under the MIT license.

If you find it useful for your research, please cite
[Donald-McCann et al. (2021)](https://arxiv.org/abs/2109.15236).
