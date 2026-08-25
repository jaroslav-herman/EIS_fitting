---
name: eec-circuit-reasoning
description: Compare, transform, or transfer parameters between impedance.py EEC circuit strings in EIS Fitting. Use for topology equivalence, arbitrary element numbering, reordered series or parallel terms, CPE parameter mapping, neighbor/batch initialization, and circuit-schema tests; do not use for numerical fitting alone.
---

# EEC circuit reasoning

Treat element suffixes as identifiers, not physical topology. Use the repository's structural helpers instead of string equality or ad hoc regular expressions.

## Use the established API

- `parse_circuit(circuit)`: parse series (`-`) and parallel (`p(...)`) hierarchy.
- `canonical_circuit(circuit)`: remove element numbering and normalize child order while retaining series/parallel structure.
- `circuits_equivalent(first, second)`: safe yes/no comparison; invalid syntax falls back to normalized exact text.
- `parameter_name_mapping(source, target)`: source element name → target element name for equivalent valid circuits, otherwise `None`.
- `map_parameter_name(parameter, mapping)`: transfer scalar and compound names such as `CPE1_0` → `CPE3_0`.

Do not compare sets of element names: `R0-p(R1,CPE1)-p(R2,CPE2)` is not equivalent to a different nesting with the same tokens. Do not zip parameter arrays by position after reordering a circuit.

## Parameter transfer

1. Establish structural equivalence and obtain the mapping.
2. Build a source dictionary keyed by mapped target parameter name.
3. Require every target parameter exactly once and verify fitted-vector length before mutation.
4. Copy `ParameterValue` objects when ownership changes; do not alias mutable parameter lists across cycles.
5. Preserve target names/units/bounds/fixed flags unless the requested operation explicitly transfers settings.
6. Clear a target's stale fit when only initial values are transferred. If block labels are deliberately swapped, reorder both parameter objects and fitted values together.

For RCPE branches, `_0` is Q and `_1` is the exponent alpha. When ordering physical processes, calculate tau with the existing `wepy.eis.tau`/`cpe_tau` convention and move the whole `(R, Q, alpha)` block together.

## Scope and tests

Improve `circuit_structure.py` when the grammar/equivalence rule is general. Keep workflow-specific choices in the caller. Add positive tests for renumbering and reordered series/parallel terms, negative tests for different hierarchy/element types, and mapping tests for compound names. Also test the consuming operation so a correct mapping cannot still corrupt fit arrays or bounds.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_circuit_structure tests.test_batch_stop -v
```
