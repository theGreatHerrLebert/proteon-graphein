# proteon-graphein: EVIDENT Case Summary

Source repo: `.` (this directory)

## Problem

`proteon-graphein` is a thin adapter that attaches `proteon`-computed
structural features (per-residue SASA, relative SASA, 8-state DSSP, and
CHARMM19+EEF1 / AMBER96 total energy) as node and graph attributes on a
[Graphein](https://github.com/a-r-j/graphein) protein graph.

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
keys. The trust question is therefore narrow: **do the values that land on the
graph equal the values `proteon` returned for the same residue?**

Because the oracle and the implementation share a process and a `proteon`
version, the tolerance is exact equality. Any drift means the adapter mutated
a value during transport, not that `proteon` is non-deterministic.

## Evidence

- `tests/test_parity.py` — three parity tests covering:
  - per-node SASA / RSA / DSSP equality (NaN-aware)
  - graph-level energy dict equality (key set + per-key value equality)
  - HETATM residues never carry DSSP codes
- Fixtures: `1crn.pdb` (pure protein) and `1ake.pdb` (protein + waters/ligand),
  exercised through `Graphein` rather than mocked.
- A guard rail (`n_checked > 0`) ensures that a future Graphein node-id
  schema change cannot make the parity test trivially pass.

## Assumptions

- The adapter runs in the same process as the oracle; `proteon` version skew
  is not in scope for this manifest.
- Graphein's default residue-level config exposes `chain_id` and
  `residue_number`; insertion codes are normalized to `""` when absent.
- Test fixtures live at `/scratch/TMAlign/proteon/test-pdbs/` and tests skip
  when fixtures are missing rather than fail.

## Failure Modes

- Graphein silently changes its node-id schema, breaking the residue-key
  match while leaving the parity test trivially passing — caught by the
  `n_checked > 0` guard.
- `proteon` adds a new energy component that the integration test does not
  check explicitly — caught by set-equality of energy dict keys.
- HETATM residues acquire DSSP codes due to a Graphein/`proteon` residue
  classification disagreement — caught on the `1ake` fixture.

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
