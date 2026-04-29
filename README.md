# proteon-graphein

[Graphein](https://github.com/a-r-j/graphein) integration for
[proteon](https://github.com/theGreatHerrLebert/proteon) — adds per-residue
SASA, relative SASA, 8-state DSSP, and CHARMM19+EEF1 / AMBER96 force-field
energies as node and graph attributes on Graphein protein graphs.

## Status

Pre-release (v0). Residue-level graph happy path with explicit
`(chain_id, residue_number, insertion_code)` key matching against
proteon's residue list. Smoke-tested against a Graphein-built graph
from `1crn.pdb`. No PyPI release yet.

## Install (development)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[graphein,dev]
```

## Usage

```python
import graphein.protein as gp
from proteon_graphein import add_proteon_features

graph = gp.construct_graph(pdb_path="1crn.pdb")
graph = add_proteon_features(graph, pdb_path="1crn.pdb")

for node, data in graph.nodes(data=True):
    print(node, data["residue_sasa"], data["rsa"], data["dssp"])

print(graph.graph["proteon_energy"]["total"], graph.graph["proteon_ff"])
```

If you only want the raw features without touching a graph:

```python
from proteon_graphein import compute_proteon_features

feats = compute_proteon_features("1crn.pdb")
print(feats["dssp"])             # 'CCCSSHHHHHHHHHHHCCC...'
print(feats["residue_sasa"])     # numpy array, Å² per residue
print(feats["energy"]["total"])  # CHARMM19+EEF1 total in kJ/mol
```

## Why

Graphein is the de facto featurization layer for protein graphs in PyTorch
Geometric / DGL workflows but does not ship a built-in physics-aware feature
set. proteon computes SASA / DSSP / force-field energies in Rust at scale
(50K PDBs in 3.5h on a single RTX 5090) and exports framework-agnostic
NumPy / Arrow outputs. This package is the thin adapter that lets a Graphein
user opt into proteon features in one call.

## Roadmap

- v0.0.1 (current): residue-level graph, SASA + RSA + DSSP + total energy,
  explicit `(chain_id, residue_number, insertion_code)` key matching, real
  Graphein integration test against `1crn.pdb`.
- v0.1.x: per-residue energy decomposition as node attributes; atom-level
  graph support.
- v0.2.x: optional batch helper for many graphs in one proteon call.
- v0.3.x: PyTorch Geometric `transform` adapter so the same features land
  inside a PyG `InMemoryDataset` pipeline without going through Graphein.

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
