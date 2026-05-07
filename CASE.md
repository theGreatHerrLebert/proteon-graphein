# proteon-graphein: EVIDENT Case Summary

Source repo: `.` (this directory)

## Problem

`proteon-graphein` is a thin adapter that attaches `proteon`-computed
structural features (per-residue SASA, relative SASA, 8-state DSSP,
backbone H-bond counts, phi/psi/omega dihedrals, per-atom SASA + atom
metadata, and CHARMM19+EEF1 / AMBER96 total energy) as node and graph
attributes on a [Graphein](https://github.com/a-r-j/graphein) protein
graph at either residue or atom granularity.

Adapters are a quiet failure surface in geometric-DL pipelines: numbers go in,
named features come out, and downstream models have no way to tell whether
the adapter preserved the source values or silently rounded, reordered, or
mismatched residues. Every demo in the geometric-DL roadmap (ProteinMPNN
refold-QC, TM-similarity dataset, Chroma conditioner, DiffDock reranking)
inherits any bias this adapter introduces.

## Trust Strategy

Validation, with `proteon` itself as the oracle.

The adapter does not *compute* features. It calls `proteon` and then routes
those values to Graphein nodes by `(chain_id, residue_number, insertion_code)`
keys at residue level — and additionally by `atom_name` for atom-level
graphs. The trust question is therefore narrow: **do the values that land on
the graph equal the values `proteon` returned for the same residue or atom?**

Because the oracle and the implementation share a process and a `proteon`
version, the tolerance is exact equality. Any drift means the adapter mutated
a value during transport, not that `proteon` is non-deterministic.

## Evidence

- `tests/test_parity.py` — five parity tests covering:
  - residue-level per-node SASA / RSA / DSSP / hbond_count / phi/psi/omega
    equality (NaN-aware: NaN positions must be absent on the graph, not
    attached as `float('nan')`)
  - atom-level per-atom SASA / charge / is_backbone / hetero equality, and
    broadcast residue features at atom granularity (all three dihedrals)
  - graph-level energy dict equality (key set + per-key value equality)
  - HETATM residues never carry DSSP codes
  - `add_proteon_features_batch(graphs, paths)` produces graphs with
    exactly the same node and graph attributes as a Python loop calling
    `add_proteon_features(graphs[i], paths[i])` one at a time, on a
    heterogeneous batch (residue + atom + residue, mixed PDBs)
- Fixtures: `1crn.pdb` (pure protein) and `1ake.pdb` (protein + waters/ligand),
  exercised through `Graphein` rather than mocked, at both residue and atom
  granularity.
- Guard rails: `n_checked > 0` and `n_atom_sasa > 0` ensure that a future
  Graphein node-id schema change cannot make the parity tests trivially pass.

## Assumptions

- The adapter runs in the same process as the oracle; `proteon` version skew
  is not in scope for this manifest.
- Graphein's default residue-level config exposes `chain_id` and
  `residue_number`; insertion codes are normalized to `""` when absent.
- Test fixtures live at `/scratch/TMAlign/proteon/test-pdbs/` and tests skip
  when fixtures are missing rather than fail.

## Failure Modes

- Graphein silently changes its node-id schema (residue or atom level),
  breaking the key match while leaving the parity test trivially passing —
  caught by the `n_checked > 0` and `n_atom_sasa > 0` guards.
- `proteon` adds a new energy component that the integration test does not
  check explicitly — caught by set-equality of energy dict keys.
- HETATM residues acquire DSSP codes due to a Graphein/`proteon` residue
  classification disagreement — caught on the `1ake` fixture.
- proteon residue/atom iteration order diverges from the flat order of
  `atom_sasa` or `backbone_dihedrals` arrays — caught because the atom-level
  parity oracle is built by walking residues → `residue.atoms` with the
  same flat counter the adapter uses.
- A future proteon batch primitive drifts from its single-call sibling in
  iteration order or rounding — caught by the batch-vs-loop full-attribute
  equality test, which compares every node and graph attribute on a
  heterogeneous batch.

## What Is Still Lacking (Deferred Claims)

The science claim of the geometric-DL roadmap — that the proteon feature pack
*improves* a published baseline on a held-out ProteinShake fold — is not
declared as an EVIDENT claim yet. A claim without a pre-declared tolerance and
decision rule would be Validation Theater. That work belongs in a separate
notebook with its own `evident.yaml` once the parity floor below is locked in.

## EVIDENT Lessons

- An adapter is a real claim surface, not glue code. The smallest meaningful
  EVIDENT artifact for an integration package is a parity manifest.
- Exact-equality oracles are cheap when the oracle and the implementation
  share a process. Use them before reaching for tolerances.
- A "no nodes matched" guard is a load-bearing test, not a paranoia check —
  Graphein's node schema is the single most likely thing to silently break
  this adapter.
