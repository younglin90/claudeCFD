import pytest

from breakup_model import DEFAULT_CHARGED_RAYLEIGH_PLATEAU_BREAKUP, charged_rayleigh_plateau_droplet_diameter


def test_charged_rayleigh_plateau_breakup_model_uses_one_global_ratio() -> None:
    model = DEFAULT_CHARGED_RAYLEIGH_PLATEAU_BREAKUP

    assert model.droplet_to_jet_ratio == pytest.approx(1.6928437314973355)
    assert charged_rayleigh_plateau_droplet_diameter(6.327633914430798e-6) == pytest.approx(1.0711695407254123e-5)
    assert model.droplet_charge_to_mass_ratio(0.62) == pytest.approx(0.62)


def test_charged_rayleigh_plateau_breakup_model_rejects_nonphysical_inputs() -> None:
    model = DEFAULT_CHARGED_RAYLEIGH_PLATEAU_BREAKUP

    with pytest.raises(ValueError, match="jet_diameter"):
        model.droplet_diameter(0.0)
    with pytest.raises(ValueError, match="charge_to_mass"):
        model.droplet_charge_to_mass_ratio(-1.0)
