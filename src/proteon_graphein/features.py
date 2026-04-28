"""Compute proteon structural features and attach them to Graphein graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import proteon

if TYPE_CHECKING:
    import networkx as nx


def compute_proteon_features(
    pdb_path: str | Path,
    *,
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    ff: str = "charmm19_eef1",
    sasa_radii: str = "bondi",
) -> dict[str, Any]:
    """Compute proteon structural features for one PDB.

    Returns a dict with these keys when their flag is enabled:
        residue_sasa : np.ndarray of per-residue SASA (Å²)
        rsa          : np.ndarray of relative SASA (NaN for non-standard residues)
        dssp         : str, one 8-state DSSP code per amino-acid residue
        energy       : dict from proteon.compute_energy (total + components)
        structure    : the loaded proteon.Structure (always populated)
    """
    structure = proteon.load(str(pdb_path))
    out: dict[str, Any] = {"structure": structure}

    if sasa:
        out["residue_sasa"] = proteon.residue_sasa(structure, radii=sasa_radii)
        out["rsa"] = proteon.relative_sasa(structure, radii=sasa_radii)

    if dssp:
        out["dssp"] = proteon.dssp(structure)

    if energy:
        out["energy"] = proteon.compute_energy(structure, ff=ff)

    return out


def add_proteon_features(
    graph: nx.Graph,
    pdb_path: str | Path,
    *,
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    ff: str = "charmm19_eef1",
) -> nx.Graph:
    """Attach proteon features to a Graphein residue-level protein graph.

    Per-node attributes added when enabled:
        residue_sasa : float (Å²)
        rsa          : float (relative SASA; may exceed 1.0 at termini, NaN for non-standard)
        dssp         : single-character 8-state DSSP code

    Graph-level attributes:
        proteon_energy : dict from proteon.compute_energy
        proteon_ff     : force-field name used

    v0 limitation:
        Matches proteon's residue order to graph nodes by sorting on
        (chain_id, residue_number). Assumes the graph was built from the
        same PDB. A residue-count mismatch raises ValueError.
    """
    feats = compute_proteon_features(
        pdb_path,
        sasa=sasa,
        dssp=dssp,
        energy=energy,
        ff=ff,
    )

    if energy:
        graph.graph["proteon_energy"] = feats["energy"]
        graph.graph["proteon_ff"] = ff

    if not (sasa or dssp):
        return graph

    nodes = sorted(
        graph.nodes(data=True),
        key=lambda n: (n[1].get("chain_id", ""), n[1].get("residue_number", 0)),
    )

    if sasa:
        sasa_vals = feats["residue_sasa"]
        rsa_vals = feats["rsa"]
        if len(sasa_vals) != len(nodes):
            raise ValueError(
                f"Residue-count mismatch: graph has {len(nodes)} nodes, "
                f"proteon found {len(sasa_vals)}. Build the graph from the same PDB."
            )
        for (node_id, _data), s, r in zip(nodes, sasa_vals, rsa_vals):
            graph.nodes[node_id]["residue_sasa"] = float(s)
            graph.nodes[node_id]["rsa"] = float(r)

    if dssp:
        ss = feats["dssp"]
        if len(ss) != len(nodes):
            raise ValueError(
                f"Residue-count mismatch (DSSP): graph has {len(nodes)} nodes, "
                f"proteon DSSP returned {len(ss)} codes."
            )
        for (node_id, _data), code in zip(nodes, ss):
            graph.nodes[node_id]["dssp"] = code

    return graph
