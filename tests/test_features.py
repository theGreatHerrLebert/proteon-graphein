"""Smoke tests for proteon-graphein."""

from __future__ import annotations

from pathlib import Path

import pytest

from proteon_graphein import compute_proteon_features


TEST_PDB = Path("/scratch/TMAlign/proteon/test-pdbs/1crn.pdb")


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_compute_features_all_flags_on() -> None:
    feats = compute_proteon_features(TEST_PDB)
    assert "residue_sasa" in feats
    assert "rsa" in feats
    assert "dssp" in feats
    assert "energy" in feats
    assert "structure" in feats

    assert len(feats["residue_sasa"]) == len(feats["dssp"]), (
        "residue_sasa and dssp must agree on residue count"
    )
    assert "total" in feats["energy"]


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_compute_features_selective_flags() -> None:
    feats = compute_proteon_features(TEST_PDB, sasa=False, dssp=True, energy=False)
    assert "residue_sasa" not in feats
    assert "rsa" not in feats
    assert "dssp" in feats
    assert "energy" not in feats


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_compute_features_amber96() -> None:
    feats = compute_proteon_features(
        TEST_PDB, sasa=False, dssp=False, energy=True, ff="amber96"
    )
    assert "total" in feats["energy"]


def test_add_proteon_features_requires_networkx() -> None:
    """The graph-attach path needs networkx + graphein; integration test is a TODO."""
    pytest.importorskip("networkx")
    pytest.importorskip("graphein")
    pytest.skip("graphein integration test pending — see TODO in README")
