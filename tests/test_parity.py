"""EVIDENT parity claim: Graphein node attributes equal direct proteon outputs.

Manifest: ../evident.yaml#proteon-graphein-residue-feature-parity

The science claim of this package is "proteon residue features attached as
Graphein node/graph attributes are the same values proteon would return when
called directly." If that doesn't hold, every downstream demo in the geometric
DL roadmap built on this adapter inherits a silent bias.

Oracle: proteon itself (compute_proteon_features), called on the same PDB.
Tolerance: exact equality for SASA/RSA/energy floats (same backend, same
process, no recomputation). String equality for DSSP codes. NaN-aware (NaN
positions in proteon's RSA must remain NaN in the graph -- and they should
simply not be attached, since float(nan) is not a useful node attribute -- so
NaN positions are tolerated by being absent rather than wrong).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from proteon_graphein import add_proteon_features, compute_proteon_features
from proteon_graphein.features import _residue_key

TEST_PDB = Path("/scratch/TMAlign/proteon/test-pdbs/1crn.pdb")
TEST_PDB_HET = Path("/scratch/TMAlign/proteon/test-pdbs/1ake.pdb")


def _build_graph(pdb_path: Path):
    pytest.importorskip("graphein")
    pytest.importorskip("networkx")
    import graphein.protein as gp
    from graphein.protein.config import ProteinGraphConfig

    return gp.construct_graph(config=ProteinGraphConfig(), path=str(pdb_path))


def _residue_index_lookups(structure):
    """Map residue key -> (index in residues, index among AA-only residues)."""
    res_idx: dict[tuple, int] = {}
    aa_idx: dict[tuple, int] = {}
    aa_counter = 0
    for i, residue in enumerate(structure.residues):
        key = _residue_key(residue.chain_id, residue.serial_number, residue.insertion_code)
        res_idx[key] = i
        if residue.is_amino_acid:
            aa_idx[key] = aa_counter
            aa_counter += 1
    return res_idx, aa_idx


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_parity_residue_features_match_direct_proteon_call() -> None:
    """Every attached node value must equal the corresponding direct-API value."""
    feats = compute_proteon_features(TEST_PDB)
    res_idx, aa_idx = _residue_index_lookups(feats["structure"])

    graph = _build_graph(TEST_PDB)
    graph = add_proteon_features(graph, TEST_PDB)

    n_checked_sasa = 0
    n_checked_dssp = 0
    for node_id, data in graph.nodes(data=True):
        key = _residue_key(
            data.get("chain_id"),
            data.get("residue_number"),
            data.get("insertion_code") or data.get("insertion"),
        )

        if "residue_sasa" in data:
            assert key in res_idx, f"node {node_id} carries SASA but no matching residue"
            i = res_idx[key]
            expected_sasa = float(feats["residue_sasa"][i])
            assert data["residue_sasa"] == expected_sasa, (
                f"SASA mismatch at {node_id}: graph={data['residue_sasa']!r} "
                f"oracle={expected_sasa!r}"
            )
            expected_rsa = float(feats["rsa"][i])
            graph_rsa = data["rsa"]
            if math.isnan(expected_rsa):
                assert math.isnan(graph_rsa), (
                    f"RSA NaN-parity broken at {node_id}: graph={graph_rsa!r}"
                )
            else:
                assert graph_rsa == expected_rsa, (
                    f"RSA mismatch at {node_id}: graph={graph_rsa!r} oracle={expected_rsa!r}"
                )
            n_checked_sasa += 1

        if "dssp" in data:
            assert key in aa_idx, f"node {node_id} carries DSSP but residue is non-AA"
            j = aa_idx[key]
            expected_dssp = feats["dssp"][j]
            assert data["dssp"] == expected_dssp, (
                f"DSSP mismatch at {node_id}: graph={data['dssp']!r} oracle={expected_dssp!r}"
            )
            n_checked_dssp += 1

    assert n_checked_sasa > 0, "no SASA values were checked -- adapter or fixture is broken"
    assert n_checked_dssp > 0, "no DSSP codes were checked -- adapter or fixture is broken"


@pytest.mark.skipif(not TEST_PDB.exists(), reason="1crn.pdb fixture not available")
def test_parity_graph_energy_matches_direct_proteon_call() -> None:
    """Graph-level proteon_energy dict must equal the direct-API energy dict."""
    feats = compute_proteon_features(TEST_PDB)

    graph = _build_graph(TEST_PDB)
    graph = add_proteon_features(graph, TEST_PDB)

    expected = feats["energy"]
    actual = graph.graph["proteon_energy"]
    assert set(actual.keys()) == set(expected.keys()), (
        f"energy keys diverged: graph={sorted(actual)} oracle={sorted(expected)}"
    )
    for key, exp_val in expected.items():
        assert actual[key] == exp_val, (
            f"energy[{key!r}] mismatch: graph={actual[key]!r} oracle={exp_val!r}"
        )
    assert graph.graph["proteon_ff"] == "charmm19_eef1"


@pytest.mark.skipif(not TEST_PDB_HET.exists(), reason="1ake.pdb fixture not available")
def test_parity_hetatm_residues_dssp_only_for_aa() -> None:
    """On a PDB with HETATMs, no non-AA node should carry a DSSP code."""
    feats = compute_proteon_features(TEST_PDB_HET)
    _, aa_idx = _residue_index_lookups(feats["structure"])

    graph = _build_graph(TEST_PDB_HET)
    graph = add_proteon_features(graph, TEST_PDB_HET)

    for node_id, data in graph.nodes(data=True):
        if "dssp" not in data:
            continue
        key = _residue_key(
            data.get("chain_id"),
            data.get("residue_number"),
            data.get("insertion_code") or data.get("insertion"),
        )
        assert key in aa_idx, (
            f"node {node_id} has DSSP but residue is not amino acid (key={key})"
        )
