from __future__ import annotations

import pytest

from propellant import Propellant, current_per_emitter, emi_bf4_reference, mass_flow_from_volume_flow


def test_propellant_rejects_nonphysical_properties() -> None:
    with pytest.raises(ValueError, match="density"):
        Propellant("bad", density=0.0, viscosity=1.0, surface_tension=1.0, conductivity=1.0, permittivity=1.0)


def test_mass_flow_from_volume_flow_uses_density() -> None:
    prop = Propellant("p", density=1200.0, viscosity=0.01, surface_tension=0.04, conductivity=0.5, permittivity=10.0)
    assert mass_flow_from_volume_flow(prop, 2.0e-12) == pytest.approx(2.4e-9)


def test_current_per_emitter_distributes_total_current() -> None:
    assert current_per_emitter(8.0e-6, 4) == pytest.approx(2.0e-6)


def test_emi_bf4_reference_has_positive_microthruster_properties() -> None:
    prop = emi_bf4_reference()
    assert prop.name
    assert prop.density > 0.0
    assert prop.conductivity > 0.0
