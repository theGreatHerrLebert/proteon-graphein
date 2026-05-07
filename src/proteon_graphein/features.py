"""Compute proteon structural features and attach them to Graphein graphs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import proteon

if TYPE_CHECKING:
    import networkx as nx


ResidueKey = tuple[str, int, str]
AtomKey = tuple[str, int, str, str]
Granularity = Literal["auto", "residue", "atom"]


# ---------------------------------------------------------------------------
# Single-structure feature computation
# ---------------------------------------------------------------------------


def compute_proteon_features(
    pdb_path: str | Path,
    *,
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    atom_sasa: bool = False,
    hbond_count: bool = False,
    dihedrals: bool = False,
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
        atom_sasa    : np.ndarray of per-atom SASA in Å² (length n_atoms,
                       flat order matching ``structure.atoms``).
        hbond_count  : np.ndarray of per-AA-residue backbone H-bond counts.
        phi, psi, omega : np.ndarray of per-AA-residue backbone dihedrals
                       in degrees, NaN at chain termini.
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

    if atom_sasa:
        out["atom_sasa"] = proteon.atom_sasa(structure, radii=sasa_radii)

    if hbond_count:
        out["hbond_count"] = proteon.hbond_count(structure)

    if dihedrals:
        phi, psi, omega = proteon.backbone_dihedrals(structure)
        out["phi"] = phi
        out["psi"] = psi
        out["omega"] = omega

    return out


# ---------------------------------------------------------------------------
# Batch feature computation (uses proteon's parallel primitives where they
# exist; pure Python loop otherwise)
# ---------------------------------------------------------------------------


def compute_proteon_features_batch(
    pdb_paths: Sequence[str | Path],
    *,
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    atom_sasa: bool = False,
    hbond_count: bool = False,
    dihedrals: bool = False,
    ff: str = "charmm19_eef1",
    sasa_radii: str = "bondi",
    n_threads: int | None = None,
) -> list[dict[str, Any]]:
    """Compute proteon features for many PDBs using rayon-parallel primitives.

    Same per-structure result shape as :func:`compute_proteon_features`. Loads
    structures via ``proteon.batch_load`` and dispatches each enabled feature
    through its corresponding ``batch_*`` primitive (added in proteon's
    batch-primitives PR). Strict mode only — one bad PDB raises.

    Args:
        pdb_paths: Sequence of paths to PDB files.
        n_threads: Thread count for the proteon batch calls.
            ``None`` / ``-1`` / ``0`` = all cores.

    Returns:
        List of feature dicts in input order.
    """
    paths = [str(p) for p in pdb_paths]
    structures = proteon.batch_load(paths, n_threads=n_threads)

    n = len(structures)
    out: list[dict[str, Any]] = [{"structure": s} for s in structures]

    if not n:
        return out

    if sasa:
        residue_sasas = proteon.batch_residue_sasa(
            structures, radii=sasa_radii, n_threads=n_threads
        )
        rsas = proteon.batch_relative_sasa(
            structures, radii=sasa_radii, n_threads=n_threads
        )
        for i, (rs, rsa) in enumerate(zip(residue_sasas, rsas)):
            out[i]["residue_sasa"] = rs
            out[i]["rsa"] = rsa

    if dssp:
        codes = proteon.batch_dssp(structures, n_threads=n_threads)
        for i, code in enumerate(codes):
            out[i]["dssp"] = code

    if energy:
        energies = proteon.batch_compute_energy(
            structures, ff=ff, n_threads=n_threads
        )
        for i, e in enumerate(energies):
            out[i]["energy"] = e

    if atom_sasa:
        per_atom = proteon.batch_atom_sasa(
            structures, radii=sasa_radii, n_threads=n_threads
        )
        for i, a in enumerate(per_atom):
            out[i]["atom_sasa"] = a

    if hbond_count:
        counts = proteon.batch_hbond_count(structures, n_threads=n_threads)
        for i, c in enumerate(counts):
            out[i]["hbond_count"] = c

    if dihedrals:
        triples = proteon.batch_dihedrals(structures, n_threads=n_threads)
        for i, (phi, psi, omega) in enumerate(triples):
            out[i]["phi"] = phi
            out[i]["psi"] = psi
            out[i]["omega"] = omega

    return out


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------


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


def _atom_key(
    chain_id: str | None,
    residue_number: int | str | None,
    insertion_code: str | None,
    atom_name: str | None,
) -> AtomKey:
    """Stable atom identity for cross-matching proteon ↔ Graphein atom-level graphs."""
    chain, resnum, icode = _residue_key(chain_id, residue_number, insertion_code)
    return (chain, resnum, icode, str(atom_name or "").strip())


def _detect_granularity(graph: nx.Graph) -> Literal["residue", "atom"]:
    """Detect Graphein granularity.

    Primary signal: ``graph.graph["config"].granularity``. Only "atom" is
    atom-level; "centroids" is one node per residue (residue-level), and any
    atom-name value (e.g. "CA") is also residue-level.

    Fallback for hand-built graphs without a Graphein config: a residue
    holding more than one node is atom-level. Default is residue.
    """
    config = graph.graph.get("config")
    granularity = getattr(config, "granularity", None)
    if isinstance(granularity, str):
        return "atom" if granularity == "atom" else "residue"

    seen: set[tuple[str, int, str]] = set()
    for _, data in graph.nodes(data=True):
        rkey = _residue_key(
            data.get("chain_id"),
            data.get("residue_number"),
            data.get("insertion_code") or data.get("insertion"),
        )
        if rkey in seen:
            return "atom"
        seen.add(rkey)
    return "residue"


def _safe_float(value: Any) -> float | None:
    """Return float(value), or None if value is NaN. Skips NaN attrs on nodes."""
    f = float(value)
    return None if math.isnan(f) else f


# ---------------------------------------------------------------------------
# Attach: write features into a graph (shared by single-call and batch paths)
# ---------------------------------------------------------------------------


def _attach_features(
    graph: nx.Graph,
    feats: dict[str, Any],
    *,
    sasa: bool,
    dssp: bool,
    energy: bool,
    hbond_count: bool,
    dihedrals: bool,
    atom_features: bool,
    granularity: Literal["residue", "atom"],
    ff: str,
) -> nx.Graph:
    """Write proteon features from a precomputed feats dict into a graph.

    Pure attach step — does not call proteon. Used by both
    :func:`add_proteon_features` (single graph) and
    :func:`add_proteon_features_batch` (many graphs in one call).

    Raises:
        ValueError: when node-level features were requested but no node
            matched (almost always a wrong-PDB pairing).
    """
    if energy:
        graph.graph["proteon_energy"] = feats["energy"]
        graph.graph["proteon_ff"] = ff

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

    hbond_lookup: dict[ResidueKey, int] = {}
    phi_lookup: dict[ResidueKey, float] = {}
    psi_lookup: dict[ResidueKey, float] = {}
    omega_lookup: dict[ResidueKey, float] = {}
    if hbond_count or dihedrals:
        aa_residues = [r for r in residues if r.is_amino_acid]
        if hbond_count and len(feats["hbond_count"]) != len(aa_residues):
            raise RuntimeError(
                f"proteon hbond_count returned {len(feats['hbond_count'])} entries "
                f"for {len(aa_residues)} amino-acid residues — internal mismatch."
            )
        if dihedrals and len(feats["phi"]) != len(aa_residues):
            raise RuntimeError(
                f"proteon backbone_dihedrals returned {len(feats['phi'])} entries "
                f"for {len(aa_residues)} amino-acid residues — internal mismatch."
            )
        for j, residue in enumerate(aa_residues):
            key = _residue_key(residue.chain_id, residue.serial_number, residue.insertion_code)
            if hbond_count:
                hbond_lookup[key] = int(feats["hbond_count"][j])
            if dihedrals:
                phi_lookup[key] = float(feats["phi"][j])
                psi_lookup[key] = float(feats["psi"][j])
                omega_lookup[key] = float(feats["omega"][j])

    atom_sasa_lookup: dict[AtomKey, float] = {}
    atom_meta_lookup: dict[AtomKey, dict[str, Any]] = {}
    if granularity == "atom":
        per_atom = feats.get("atom_sasa") if atom_features else None
        idx = 0
        for residue in residues:
            for atom in residue.atoms:
                key = _atom_key(
                    residue.chain_id,
                    residue.serial_number,
                    residue.insertion_code,
                    atom.name,
                )
                if per_atom is not None:
                    atom_sasa_lookup[key] = float(per_atom[idx])
                if atom_features:
                    atom_meta_lookup[key] = {
                        "charge": float(atom.charge),
                        "is_backbone": bool(atom.is_backbone),
                        "hetero": bool(atom.hetero),
                    }
                idx += 1

    n_attached = 0
    for node_id, data in graph.nodes(data=True):
        rkey = _residue_key(
            data.get("chain_id"),
            data.get("residue_number"),
            data.get("insertion_code") or data.get("insertion"),
        )
        node_attrs = graph.nodes[node_id]
        attached_any = False

        if sasa and rkey in sasa_lookup:
            node_attrs["residue_sasa"] = sasa_lookup[rkey]
            rsa_val = _safe_float(rsa_lookup[rkey])
            if rsa_val is not None:
                node_attrs["rsa"] = rsa_val
            attached_any = True
        if dssp and rkey in dssp_lookup:
            node_attrs["dssp"] = dssp_lookup[rkey]
            attached_any = True
        if hbond_count and rkey in hbond_lookup:
            node_attrs["hbond_count"] = hbond_lookup[rkey]
            attached_any = True
        if dihedrals and rkey in phi_lookup:
            for name, lookup in (("phi", phi_lookup), ("psi", psi_lookup), ("omega", omega_lookup)):
                val = _safe_float(lookup[rkey])
                if val is not None:
                    node_attrs[name] = val
            attached_any = True

        if granularity == "atom":
            akey = _atom_key(
                data.get("chain_id"),
                data.get("residue_number"),
                data.get("insertion_code") or data.get("insertion"),
                data.get("atom_type") or data.get("atom_name"),
            )
            if akey in atom_sasa_lookup:
                node_attrs["atom_sasa"] = atom_sasa_lookup[akey]
                attached_any = True
            if akey in atom_meta_lookup:
                for k, v in atom_meta_lookup[akey].items():
                    node_attrs[k] = v
                attached_any = True

        if attached_any:
            n_attached += 1

    requested_node_features = sasa or dssp or hbond_count or dihedrals or (
        granularity == "atom" and atom_features
    )
    if (
        requested_node_features
        and n_attached == 0
        and graph.number_of_nodes() > 0
    ):
        raise ValueError(
            "No graph nodes matched proteon residues by "
            "(chain_id, residue_number, insertion_code). "
            "Was the graph built from the same PDB?"
        )

    return graph


# ---------------------------------------------------------------------------
# Public attach entry points
# ---------------------------------------------------------------------------


def add_proteon_features(
    graph: nx.Graph,
    pdb_path: str | Path,
    *,
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    hbond_count: bool = False,
    dihedrals: bool = False,
    atom_features: bool = True,
    granularity: Granularity = "auto",
    ff: str = "charmm19_eef1",
) -> nx.Graph:
    """Attach proteon features to a Graphein protein graph.

    Granularity is auto-detected from the graph (residue vs atom). Pass
    ``granularity="residue"`` or ``"atom"`` to override.

    Per-node attributes added when enabled:
        residue_sasa : float (Å²) per residue. On atom-level graphs, broadcast
                       to every atom of the residue.
        rsa          : relative SASA (may exceed 1.0 at termini). NaN values
                       are skipped (attribute simply absent).
        dssp         : 8-state DSSP code, AA residues only. Broadcast to atoms
                       on atom-level graphs.
        hbond_count  : backbone H-bond count per AA residue (uint). Broadcast
                       to atoms on atom-level graphs.
        phi, psi, omega : backbone dihedrals in degrees, AA residues only.
                       NaN at chain termini are skipped.
        atom_sasa    : per-atom SASA in Å² (atom-level graphs only).
        charge       : partial charge from proteon.Atom (atom-level only).
        is_backbone  : bool, atom-level only.
        hetero       : bool (HETATM flag), atom-level only.

    Graph-level attributes added when ``energy=True``:
        proteon_energy : dict from proteon.compute_energy
        proteon_ff     : the force-field name used

    Args:
        atom_features: when False on an atom-level graph, skips per-atom
            attributes (atom_sasa/charge/is_backbone/hetero) but still
            broadcasts residue features.

    Matching is by (chain_id, residue_number, insertion_code), and additionally
    by atom_name for atom-level graphs. Insertion_code is normalized to ``""``
    when missing.

    Raises:
        ValueError: when no graph node could be matched (almost always means
            the graph was built from a different PDB).
    """
    if granularity == "auto":
        granularity = _detect_granularity(graph)

    feats = compute_proteon_features(
        pdb_path,
        sasa=sasa,
        dssp=dssp,
        energy=energy,
        atom_sasa=(granularity == "atom" and atom_features),
        hbond_count=hbond_count,
        dihedrals=dihedrals,
        ff=ff,
    )

    return _attach_features(
        graph,
        feats,
        sasa=sasa,
        dssp=dssp,
        energy=energy,
        hbond_count=hbond_count,
        dihedrals=dihedrals,
        atom_features=atom_features,
        granularity=granularity,
        ff=ff,
    )


def add_proteon_features_batch(
    graphs: Sequence[nx.Graph],
    pdb_paths: Sequence[str | Path],
    *,
    sasa: bool = True,
    dssp: bool = True,
    energy: bool = True,
    hbond_count: bool = False,
    dihedrals: bool = False,
    atom_features: bool = True,
    granularity: Granularity = "auto",
    ff: str = "charmm19_eef1",
    n_threads: int | None = None,
) -> list[nx.Graph]:
    """Attach proteon features to many graphs in one batched proteon call.

    Loads structures and computes features in parallel via proteon's batch
    primitives, then applies the attach step to each (graph, feats) pair.
    Strict mode only — one bad PDB raises.

    Granularity is detected per graph (mixing residue and atom-level graphs
    in one batch is supported); when any graph is atom-level, ``batch_atom_sasa``
    runs for the whole batch, and per-atom features are written only into
    atom-level graphs.

    Args:
        graphs: Sequence of Graphein graphs in the same order as ``pdb_paths``.
        pdb_paths: Sequence of paths to the PDB files used to build ``graphs``.
        n_threads: Thread count for proteon batch calls.

    Returns:
        The same ``graphs`` list, with features attached in place.

    Raises:
        ValueError: on length mismatch, or when a graph's nodes do not match
            its paired PDB.
    """
    if len(graphs) != len(pdb_paths):
        raise ValueError(
            f"len(graphs)={len(graphs)} does not match len(pdb_paths)={len(pdb_paths)}"
        )
    if not graphs:
        return list(graphs)

    if granularity == "auto":
        per_graph_granularity: list[Literal["residue", "atom"]] = [
            _detect_granularity(g) for g in graphs
        ]
    else:
        per_graph_granularity = [granularity] * len(graphs)

    any_atom = any(g == "atom" for g in per_graph_granularity)

    feats_list = compute_proteon_features_batch(
        pdb_paths,
        sasa=sasa,
        dssp=dssp,
        energy=energy,
        atom_sasa=(any_atom and atom_features),
        hbond_count=hbond_count,
        dihedrals=dihedrals,
        ff=ff,
        n_threads=n_threads,
    )

    for graph, feats, gr in zip(graphs, feats_list, per_graph_granularity):
        _attach_features(
            graph,
            feats,
            sasa=sasa,
            dssp=dssp,
            energy=energy,
            hbond_count=hbond_count,
            dihedrals=dihedrals,
            atom_features=atom_features,
            granularity=gr,
            ff=ff,
        )

    return list(graphs)
