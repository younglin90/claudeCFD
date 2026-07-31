from pathlib import Path


def test_external_benchmark_registry_contains_required_literature_blocks() -> None:
    text = Path("docs/electrospray/external_benchmark_registry.md").read_text(encoding="utf-8")

    for required in (
        "Huh and Richard E. Wirz",
        "10.1063/5.0120737",
        "arXiv:2111.10383",
        "Debasish Das and David Saintillan",
        "10.1017/jfm.2016.704",
        "arXiv:1605.04036",
        "10.1017/jfm.2017.560",
        "arXiv:1612.02070",
        "Qiang Liu, Jie Zhang, and Jian Wu",
        "arXiv:2207.08152",
    ):
        assert required in text


def test_external_benchmark_registry_keeps_comparison_as_future_numeric_gate() -> None:
    text = Path("docs/electrospray/external_benchmark_registry.md").read_text(encoding="utf-8")

    assert "not a claim that the current reduced solver has already reproduced every external dataset" in text
    assert "numeric comparison table" in text
    assert "what the local solver can and cannot reproduce" in text
