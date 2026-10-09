# SKZ-905 macOS wheel investigation only

This diagnostic PR must not be merged as a product fix. It does not change the
product's Cargo profile or release workflows, create releases, or publish packages.
The workflow has read-only permissions and uses only pull-request events.

The local-failed wheel is an unpublished candidate built from public main at
`6c19303f4689037e618a2622819592a415b71496`, still carrying version 0.9.1 metadata.
Its SHA-256 is `446e374bb8605ee36efb1ac44f2e437080d16ba2352236b367fafdcad6838412`.
It contains no production inputs or credentials. Do not install it outside isolated
diagnostic environments. The lock fixture preserves the original dependency set.

Two ARM64 runner labels each build the same source with Rust 1.97.1 and maturin
1.15.0: original release stripping and explicitly disabled stripping. The actual
OS, compiler, SDK, linker, dependency versions, binary layout, wheel hashes,
import exit codes, and pytest outcomes are recorded rather than inferred from labels.
Original-variant jobs additionally load the identical local failed wheel.

Each wheel is installed in a separate environment. Python isolated mode and pytest
importlib mode prevent the source checkout from shadowing the installed package.
The 295-test subset covers imports, daily performance, backtesting, and input validation.

A successful diagnostic job means observations were collected and the unstripped
control passed; it does not mean every wheel loaded. Read the individual JSON
results and logs. A stripped wheel rejected at import will also run pytest and
record its collection failure, rather than report that tests passed or were skipped.
This is not full release acceptance or validation of every macOS/architecture.
