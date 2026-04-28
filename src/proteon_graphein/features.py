"""Compute proteon structural features and attach them to Graphein graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import proteon

if TYPE_CHECKING:
    import networkx as nx


ResidueKey = tuple[str, int, str]


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
        residue_sasa : np.ndarray of per-residue SASA in Å² (one entry per residue,
                       including non-AA residues like waters/ligands).
        rsa          : np.ndarray of relative SASA (NaN for non-standard residues).
        dssp         : str, one 8-state DSSP code per **amino-acid** residue.
                       Length equals ``sum(r.is_amino_acid for r in structure.residues)``.
        energy       : dict from proteon.compute_energy (total + components).
        structure    : the loaded proteon.Structure (always populated).
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


def _residue_key(
    chain_id: str | None,
    residue_number: int | str | None,
    insertion_code: str | None,
) -> ResidueKey:
    """Stable residue identity for cross-matching proteon ↔ Graphein.

    insertion_code is normalized to ``""`` when None or whitespace.
    """
    icode = "" if insertion_code is None else str(insertion_code).strip()
    return (str(chain_id or ""), int(residue_number or 0), icode)


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
        residue_sasa : float (Å²) — every node whose residue key is found.
        rsa          : float (relative SASA; may exceed 1.0 at termini, NaN for
                       non-standard residues).
        dssp         : single-character 8-state DSSP code — only attached to
                       amino-acid residues. Non-AA nodes are left unchanged.

    Graph-level attributes added when ``energy=True``:
        proteon_energy : dict from proteon.compute_energy
        proteon_ff     : the force-field name used

    Matching is by (chain_id, residue_number, insertion_code). Graphein's
    default residue-level node attrs do not include insertion_code, so it is
    treated as ``""``; structures relying on insertion codes need a Graphein
    config that surfaces them.

    Raises:
        ValueError: when no graph node could be matched to any proteon residue
            (almost always means the graph was built from a different PDB).
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

    structure = feats["structure"]
    residues = structure.residues

    sasa_lookup: dict[ResidueKey, float] = {}
    rsa_lookup: dict[ResidueKey, float] = {}
    if sasa:
        for residue, s_val, rsa_val in zip(residues, feats["residue_sasa"], feats["rsa"]):
            key = _residue_key(residue.chain_id, residue.serial_number, residue.insertion_code)
            sasa_lookup[key] = float(s_val)
            rsa_lookup[key] = float(rsa_val)

    dssp_lookup: dict[ResidueKey, str] = {}
    if dssp:
        aa_residues = [r for r in residues if r.is_amino_acid]
        dssp_str = feats["dssp"]
        if len(dssp_str) != len(aa_residues):
            raise RuntimeError(
                f"proteon DSSP returned {len(dssp_str)} codes for "
                f"{len(aa_residues)} amino-acid residues — internal mismatch."
            )
        for residue, code in zip(aa_residues, dssp_str):
            key = _residue_key(residue.chain_id, residue.serial_number, residue.insertion_code)
            dssp_lookup[key] = code

    n_attached = 0
    for node_id, data in graph.nodes(data=True):
        key = _residue_key(
            data.get("chain_id"),
            data.get("residue_number"),
            data.get("insertion_code") or data.get("insertion"),
        )
        attached_any = False
        if sasa and key in sasa_lookup:
            graph.nodes[node_id]["residue_sasa"] = sasa_lookup[key]
            graph.nodes[node_id]["rsa"] = rsa_lookup[key]
            attached_any = True
        if dssp and key in dssp_lookup:
            graph.nodes[node_id]["dssp"] = dssp_lookup[key]
            attached_any = True
        if attached_any:
            n_attached += 1

    if n_attached == 0 and graph.number_of_nodes() > 0:
        raise ValueError(
            "No graph nodes matched proteon residues by "
            "(chain_id, residue_number, insertion_code). "
            "Was the graph built from the same PDB?"
        )

    return graph
