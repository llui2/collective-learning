# Coding, repositories, and scientific figures

Use this module for code changes, numerical experiments, repository work, figures, builds, and validation.

First identify the scale of the task. The minimal scientific-script conventions below apply to research scripts and figure generation. Do not force them onto a larger software system that already has justified modules, classes, CLIs, tests, or interfaces.

## Expression and calculation scale

- Keep mathematical formulas recognizable in the code.
- Prefer direct NumPy/SciPy expressions over layers of wrappers.
- Use short names that map naturally to the mathematics.
- Keep fixed experiment parameters visible near the top of a small script.
- Do not introduce an extra computational parameter unless it carries independent information.
- Comment why a non-obvious scientific or geometric step exists; do not narrate Python syntax.
- Use explicit random generators or seeds when randomness matters.
- Add assertions only when they encode a real invariant, shape, finite-value requirement, or expected limiting behavior.

### Code notation

Mirror the paper notation without making Python unreadable.

Prefer ASCII transliterations for mathematical identifiers:
- `rho`, `rho_local`;
- `theta`, `theta_b`;
- `tau`, `tau_c`;
- `lambda2` or a descriptive alternative when `lambda` is unavailable as an identifier.

Use uppercase module constants for fixed configuration/path values in a script:

`RHO`, `THETA_B`, `RTOL`, `OUT`, `FIG_PATH`.

Use lowercase names for arrays and computed mathematical quantities:

`rho`, `theta`, `erosion`, `eigenvalues`.

A mathematical function may keep the paper's conventional name, e.g. `U(theta)`, when this makes the code easier to compare with the derivation. Descriptive helpers use snake_case.

Avoid Unicode mathematical characters in Python identifiers. Keep the symbols in LaTeX labels, captions, and manuscript equations.

## Function scale

Use a function when at least one is true:
- the operation is reused;
- it represents a named mathematical or scientific quantity;
- it is independently testable;
- keeping it inline obscures the main calculation.

Do not extract helpers merely to shorten the visible script.

For ordinary scientific scripts, prefer functions and arrays over classes. A class is justified when persistent state and behavior genuinely belong together.

## File scale

A small research script should usually read top-to-bottom:

imports -> paths/constants -> parameters -> model/data -> computation -> figure/output.

Keep the main calculation visible.

The source projects strongly favor one script per scientific artifact or figure. File names should reflect the output or pipeline role directly.

For a normal figure script:
- no CLI parser;
- no `main()` ceremony;
- no package structure;
- no configuration object;
- no generic helper layer.

Add those only when the script has real modes, user inputs, reuse as a command-line tool, or composition that benefits from an explicit entry point.

Prefer imports grouped as standard library, third-party libraries, then local project imports.

Use a short file docstring when one sentence can state the scientific purpose. Avoid large banner comments and boilerplate section headers in new small scripts.

## Compute versus plot

Keep cheap computation and plotting together when that makes the scientific logic easier to inspect.

Separate computation from plotting when:
- the simulation or sweep is expensive;
- the same canonical data feeds several figures;
- figure polishing should not rerun the numerical experiment;
- provenance of computed data matters independently.

In that case, prefer a simple pairing such as:

`fig3_data.py -> fig3_data.csv -> fig3_plot.py`.

Do not create this split for a calculation that takes seconds and is clearer inline.

## Repository scale

Treat files, dependencies, abstractions, helper layers, configuration, and total lines of code as costs.

- Prefer one readable file over several tiny files when there is no meaningful boundary.
- Reuse existing project machinery.
- Keep the root small and obvious.
- Use one canonical path per generated artifact.
- Do not create speculative directories for future work.
- Do not create duplicate outputs such as `final2`, `new_final`, or `latest_fixed`.
- Keep generated junk, caches, editor files, and temporary exports out of Git.
- For ordinary bounded work, use `main`; use a branch/worktree only when isolation is useful or explicitly requested.

When a repository provides `./build.sh`, treat it as the canonical build unless the project says otherwise.

## Dependency style

Prefer the standard scientific stack when sufficient:
- NumPy;
- SciPy;
- Matplotlib;
- NetworkX when graph operations are needed.

Use pandas when tabular operations actually benefit from it. Do not import it for trivial array/file work.

Use plain Matplotlib by default. Do not add seaborn or another plotting layer only to obtain a visual style.

## Figure work

For scientific figures, read `.agents/figures.md` in addition to this file.

Keep calculation and data-generation logic here; keep typography, labels, legends, panel layout, color, and output-style decisions in the figure module.
