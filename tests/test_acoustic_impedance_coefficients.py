"""Unit checks for 07 acoustic impedance sign conventions."""
from __future__ import annotations


def coeffs(Z1, Z2):
    Rp = (Z2 - Z1) / (Z2 + Z1)
    Tp = 2.0 * Z2 / (Z1 + Z2)
    Ru = -Rp
    Tu = 2.0 * Z1 / (Z1 + Z2)
    return Rp, Tp, Ru, Tu


def test_coefficients():
    media = {
        "air": (1.157, 347.8),
        "helium": (0.164, 1008.2),
        "argon": (1.748, 308.2),
        "water": (998.0, 1344.6),
    }
    Z = {k: rho * c for k, (rho, c) in media.items()}

    Rp, Tp, Ru, Tu = coeffs(Z["air"], Z["water"])
    assert Rp > 0.99
    assert 1.99 < Tp < 2.01
    assert Ru < -0.99
    assert 0.0 < Tu < 0.01

    Rp, Tp, Ru, Tu = coeffs(Z["helium"], Z["air"])
    assert 0.40 < Rp < 0.43
    assert 1.40 < Tp < 1.44
    assert -0.43 < Ru < -0.40
    assert 0.56 < Tu < 0.60

    Rp, Tp, Ru, Tu = coeffs(Z["argon"], Z["air"])
    assert -0.16 < Rp < -0.13
    assert 0.84 < Tp < 0.87
    assert 0.13 < Ru < 0.16
    assert 1.13 < Tu < 1.16


if __name__ == "__main__":
    test_coefficients()
    print("test_acoustic_impedance_coefficients: PASS")

