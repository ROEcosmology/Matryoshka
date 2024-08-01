"""EFT-of-LSS power spectrum multipoles."""
import numpy as np


def multipole(P_n, b, f, stochastic=None, kbins=None, ng=None, multipole=None):
    """
    Calculates the galaxy power spectrum multipole given a `P_n` matrix
    that corresponds to the desired multipole.

    Args:
        P_n (list of array): Power spectrum components.
            The arrays correspond to tree level, 1-loop and counterterm
            components and should have shape ``(3, nk)``, ``(12, nk)``
            and ``(6, nk)`` respectively, where ``nk`` is the number
            of wavenumber bins.
        b (array): Bias parameters and counter terms.
            Should have length 7.
        f (float): Growth rate at the same redshift as `P_n`.
        stochastic (array) : Stochastic counterterms.
            Should have length 3. Default is ``None``, in which case no
            stochastic terms are included.
        kbins (array): wavenumber bins associated with ``P_n``.
            Only required if ``stochastic`` is not ``None``. Default is ``None``.
        ng (float): Mean galaxy number density.
            Only required if ``stochastic`` is not ``None``. Default is ``None``.
        multipole (int): Desired multipole.
            Only required if ``stochastic`` is not ``None``.
            Can either be 0 or 2. Default is ``None``.

    Returns:
        1-d array: The galaxy power spectrum multipole.
    """
    # The block of code is a slightly modified version of
    # the code in cell 21 of the example PyBird notebook run_pybird.ipynb
    b1, b2, b3, b4, b5, b6, b7 = b
    b11 = np.array([
        b1**2,
        2.0 * b1 * f,
        f**2,
    ])
    bloop = np.array([
        1.0,
        b1,
        b2,
        b3,
        b4,
        b1 * b1,
        b1 * b2,
        b1 * b3,
        b1 * b4,
        b2 * b2,
        b2 * b4,
        b4 * b4,
    ])
    bct = 2.0 * np.array([
        b1 * b5,
        b1 * b6,
        b1 * b7,
        b5 * f,
        b6 * f,
        b7 * f,
    ])

    Plin = np.einsum("b,bx->x", b11, P_n[0])
    Ploop = np.einsum("b,bx->x", bloop, P_n[1])
    Pct = np.einsum("b,bx->x", bct, P_n[2])

    Pk = Plin + Ploop + Pct

    if stochastic is None:
        return Pk
    if multipole == 0:
        return Pk + stochastic[0] / ng + (stochastic[1] * kbins**2) / ng
    if multipole == 2:
        return Pk + (stochastic[2] * kbins**2) / ng


def multipole_vec(P_n, b, f, stochastic=None, kbins=None, ng=None, multipole=None):
    """
    Vectorized :func:`multipole` that allows for the galaxy
    power spectrum multipoles to be calculated for multiple cosmologies.

    Args:
        P_n (list of array) : Power spectrum components.
            The arrays correspond to tree level, 1-loop and counterterm
            components and should have shape ``(nd, 3, nk)``,
            ``(nd, 12, nk)`` and ``(nd, 6, nk)`` respectively, where
            ``nk`` is the number of wavenumber bins and ``nd`` is the
            number of cosmologies.
        b (array) : Array of bias parameters and counter terms.
            Should have shape ``(nd, 7)``.
        f (array) : Growth rate at the same redshift as `P_n`.
            Should have shape ``(nd, 1)``.
        stochastic (array) :
            Should have shape ``(nd, 3)``. Default is ``None``,
            in which case no stochastic terms are included.
        kbins (array): wavenumber bins associated with ``P_n``.
            Only required if ``stochastic`` is not ``None``. Default is ``None``.
        ng (float): Mean galaxy number density.
            Only required if ``stochastic`` is not ``None``. Default is ``None``.
        multipole (int): Desired multipole.
            Only required if ``stochastic`` is not ``None``.
            Can either be 0 or 2. Default is ``None``.

    Returns:
        array: The galaxy power spectrum multipoles.
    """
    # The block of code is a slightly modified version of
    # the code in cell 21 of the example PyBird notebook run_pybird.ipynb
    b1, b2, b3, b4, b5, b6, b7 = np.split(b, 7, axis=1)
    b11 = np.array([
        b1**2,
        2.0 * b1 * f,
        f**2
    ])[:, :, 0].T
    bloop = np.array([
        np.ones((len(b), 1)),
        b1,
        b2,
        b3,
        b4,
        b1 * b1,
        b1 * b2,
        b1 * b3,
        b1 * b4,
        b2 * b2,
        b2 * b4,
        b4 * b4,
    ])[:, :, 0].T
    bct = 2.0 * np.array([
        b1 * b5,
        b1 * b6,
        b1 * b7,
        b5 * f,
        b6 * f,
        b7 * f,
    ])[:, :, 0].T

    Plin = np.einsum("nb,nbx->nx", b11, P_n[0])
    Ploop = np.einsum("nb,nbx->nx", bloop, P_n[1])
    Pct = np.einsum("nb,nbx->nx", bct, P_n[2])

    Pk = Plin + Ploop + Pct

    if stochastic is None:
        return Pk
    if multipole == 0:
        return (
            Pk
            + stochastic[:, 0].reshape(-1, 1) / ng
            + (stochastic[:, 1].reshape(-1, 1) * kbins**2) / ng
        )
    if multipole == 2:
        return Pk + (stochastic[:, 2].reshape(-1, 1) * kbins**2) / ng
