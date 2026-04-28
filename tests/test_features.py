"""Tests for proteon-graphein."""

from __future__ import annotations

from pathlib import Path

import pytest

from proteon_graphein import (
    add_proteon_features,
    compute_proteon_features,
)
from proteon_graphein.features import _residue_key


TEST_PDB = Path("/scratch/TMAlign/proteon/test-pdbs/1crn.pdb")
TEST_PDB_HET = Path("/scratch/TMAlign/proteon/test-pdbs/1ake.pdb")


# ---------------------------------------------------------------------------
# compute_proteon_features
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_compute_features_all_flags_on() -> None:
    feats = compute_proteon_features(TEST_PDB)
    assert "residue_sasa" in feats
    assert "rsa" in feats
    assert "dssp" in feats
    assert "energy" in feats
    assert "structure" in feats
    assert len(feats["residue_sasa"]) == len(feats["dssp"]), (
        "1crn is pure protein; SASA and DSSP residue counts must match"
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


@pytest.mark.skipif(not TEST_PDB_HET.exists(), reason="1ake.pdb fixture not available")
def test_compute_features_handles_hetatm() -> None:
    """SASA covers all residues including HETATM; DSSP only covers AA."""
    feats = compute_proteon_features(TEST_PDB_HET)
    n_sasa = len(feats["residue_sasa"])
    n_dssp = len(feats["dssp"])
    assert n_sasa > n_dssp, "1ake has waters/ligand — SASA should exceed DSSP count"


# ---------------------------------------------------------------------------
# _residue_key
# ---------------------------------------------------------------------------


def test_residue_key_normalizes_none_icode() -> None:
    assert _residue_key("A", 1, None) == ("A", 1, "")
    assert _residue_key("A", 1, "") == ("A", 1, "")
    assert _residue_key("A", 1, " ") == ("A", 1, "")
    assert _residue_key("A", 1, "B") == ("A", 1, "B")


# ---------------------------------------------------------------------------
# add_proteon_features (Graphein integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_add_features_to_graphein_graph() -> None:
    pytest.importorskip("graphein")
    pytest.importorskip("networkx")
    import graphein.protein as gp
    from graphein.protein.config import ProteinGraphConfig

    config = ProteinGraphConfig()
    graph = gp.construct_graph(config=config, path=str(TEST_PDB))
    n_nodes_before = graph.number_of_nodes()

    graph = add_proteon_features(graph, TEST_PDB)

    assert graph.number_of_nodes() == n_nodes_before, "no nodes added/removed"
    assert "proteon_energy" in graph.graph
    assert "proteon_ff" in graph.graph
    assert graph.graph["proteon_ff"] == "charmm19_eef1"
    assert "total" in graph.graph["proteon_energy"]

    for node_id, data in graph.nodes(data=True):
        assert "residue_sasa" in data, f"node {node_id} missing residue_sasa"
        assert "rsa" in data, f"node {node_id} missing rsa"
        assert "dssp" in data, f"node {node_id} missing dssp"
        assert isinstance(data["residue_sasa"], float)
        assert isinstance(data["dssp"], str) and len(data["dssp"]) == 1
        assert data["residue_sasa"] >= 0.0


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_add_features_selective_flags() -> None:
    pytest.importorskip("graphein")
    import graphein.protein as gp
    from graphein.protein.config import ProteinGraphConfig

    graph = gp.construct_graph(config=ProteinGraphConfig(), path=str(TEST_PDB))
    graph = add_proteon_features(graph, TEST_PDB, sasa=True, dssp=False, energy=False)

    assert "proteon_energy" not in graph.graph
    sample = next(iter(graph.nodes(data=True)))[1]
    assert "residue_sasa" in sample
    assert "rsa" in sample
    assert "dssp" not in sample


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_add_features_mismatched_pdb_raises() -> None:
    """Building a graph from one PDB and feeding a different one should raise."""
    pytest.importorskip("graphein")
    import networkx as nx

    # Hand-build a graph whose node keys won't match anything in 1crn
    graph = nx.Graph()
    graph.add_node("Z:XXX:9999", chain_id="Z", residue_number=9999, residue_name="XXX")

    with pytest.raises(ValueError, match="No graph nodes matched"):
        add_proteon_features(graph, TEST_PDB, energy=False)
