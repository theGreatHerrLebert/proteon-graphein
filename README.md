# proteon-graphein

[Graphein](https://github.com/a-r-j/graphein) integration for
[proteon](https://github.com/theGreatHerrLebert/proteon) — adds per-residue
SASA, relative SASA, 8-state DSSP, backbone H-bond counts, phi/psi/omega
dihedrals, per-atom SASA + atom metadata, and CHARMM19+EEF1 / AMBER96
force-field energies as node and graph attributes on Graphein protein
graphs at either residue or atom granularity.

## Status

Pre-release (v0.2). Residue and atom-level graphs supported with
`(chain_id, residue_number, insertion_code[, atom_name])` key matching
against proteon's residue/atom list. Granularity is auto-detected from
the graph. v0.2 adds a batch entry point that goes through proteon's
rayon-parallel primitives (`batch_residue_sasa`, `batch_relative_sasa`,
`batch_atom_sasa`, `batch_dssp`, `batch_dihedrals`, `batch_hbond_count`,
`batch_compute_energy`). Parity-tested against Graphein-built graphs from
`1crn.pdb` and `1ake.pdb`, both residue and atom granularity, including
batch-vs-loop equality. No PyPI release yet.

## Install (development)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[graphein,dev]
```

## Usage

Residue-level graph (Graphein default — CA atoms):

```python
import graphein.protein as gp
from proteon_graphein import add_proteon_features

graph = gp.construct_graph(path="1crn.pdb")
graph = add_proteon_features(graph, "1crn.pdb", hbond_count=True, dihedrals=True)

for node, data in graph.nodes(data=True):
    print(node, data["residue_sasa"], data["dssp"], data.get("phi"))

print(graph.graph["proteon_energy"]["total"], graph.graph["proteon_ff"])
```

Atom-level graph — granularity is auto-detected:

```python
from graphein.protein.config import ProteinGraphConfig

graph = gp.construct_graph(config=ProteinGraphConfig(granularity="atom"), path="1crn.pdb")
graph = add_proteon_features(graph, "1crn.pdb", hbond_count=True, dihedrals=True)

sample = next(iter(graph.nodes(data=True)))[1]
print(sample["atom_sasa"], sample["charge"], sample["is_backbone"])  # per-atom
print(sample["residue_sasa"], sample["dssp"], sample["hbond_count"])  # broadcast
```

If you only want the raw features without touching a graph:

```python
from proteon_graphein import compute_proteon_features

feats = compute_proteon_features(
    "1crn.pdb", atom_sasa=True, hbond_count=True, dihedrals=True
)
print(feats["dssp"])             # 'CCCSSHHHHHHHHHHHCCC...'
print(feats["residue_sasa"])     # numpy array, Å² per residue
print(feats["atom_sasa"])        # numpy array, Å² per atom
print(feats["phi"])              # degrees, NaN at chain termini
print(feats["energy"]["total"])  # CHARMM19+EEF1 total in kJ/mol
```

Batched over many PDBs in one parallel proteon call:

```python
from proteon_graphein import add_proteon_features_batch

graphs = [gp.construct_graph(path=p) for p in pdb_paths]
add_proteon_features_batch(
    graphs, pdb_paths,
    hbond_count=True, dihedrals=True,
    n_threads=-1,  # all cores
)
```

Each graph in `graphs` ends up with the same attributes as a single-call
`add_proteon_features` but loading and feature compute happen in parallel
in Rust. Graphs with different granularity can be mixed in one batch.
Strict mode only — one bad PDB raises.

## Why

Graphein is the de facto featurization layer for protein graphs in PyTorch
Geometric / DGL workflows but does not ship a built-in physics-aware feature
set. proteon computes SASA / DSSP / force-field energies in Rust at scale
(50K PDBs in 3.5h on a single RTX 5090) and exports framework-agnostic
NumPy / Arrow outputs. This package is the thin adapter that lets a Graphein
user opt into proteon features in one call.

## Roadmap

- v0.0.1: residue-level graph, SASA + RSA + DSSP + total energy, explicit
  `(chain_id, residue_number, insertion_code)` key matching, real Graphein
  integration test against `1crn.pdb`.
- v0.1.0: atom-level graph support with auto-detected granularity,
  per-atom SASA + charge + is_backbone + hetero, residue features broadcast
  to atoms, plus new residue-level features (hbond_count, phi/psi/omega).
  Atom-level parity claim added to the EVIDENT manifest.
- v0.2.0 (current): `add_proteon_features_batch` and
  `compute_proteon_features_batch` — load + compute features for many PDBs
  in one parallel proteon call via batch primitives (added upstream in
  proteon for this release). Strict mode only. Mixed-granularity batches
  supported. Batch-equals-loop parity claim added to the EVIDENT manifest.
- v0.3.x: PyTorch Geometric `transform` adapter so the same features land
  inside a PyG `InMemoryDataset` pipeline without going through Graphein.
- v0.2.x follow-ups: tolerant batch loading (skip + return success
  indices); accepting pre-loaded `proteon.Structure` objects directly.
- Deferred: per-residue / per-atom energy decomposition. Blocked on a
  proteon API for residue-resolved energy components — `compute_energy`
  currently returns whole-structure totals only.

## Trust

This package follows the
[EVIDENT](https://github.com/theGreatHerrLebert/evident) claim-based evidence
workflow. Its current trust claim is documented in
[`evident.yaml`](evident.yaml) and explained in [`CASE.md`](CASE.md):

- **Parity claim** (tier `ci`): every value the adapter attaches to a
  Graphein node is exactly equal to what `proteon` returned for that residue.
  Reproduced by `pytest tests/test_parity.py`.
- The downstream science claim (proteon features improve geometric-DL
  baselines on a held-out fold) is tracked under `deferred_claims:` and will
  only move into `claims:` once a tolerance and decision rule are pre-declared
  — declaring it earlier would be the [Validation Theater](https://github.com/theGreatHerrLebert/evident/blob/main/anti-patterns/README.md)
  anti-pattern.

To validate the manifest structurally:

```bash
python /path/to/evident/workflow/validate_manifest.py evident.yaml
```

## License

MIT — see [LICENSE](LICENSE).
