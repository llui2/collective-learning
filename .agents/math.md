# Mathematical reasoning and notation

Use this module for derivations, notation, analytical arguments, approximations, spectral reasoning, and mathematical interpretation.

## Core rules

- Start from the concrete mathematical, physical, network, or geometric object.
- Introduce notation only when it carries information needed later.
- Use the smallest useful set of symbols.
- Do not introduce an extra parameter unless it carries independent information.
- Keep each symbol's meaning stable.
- State assumptions before they are used.
- Move through one nontrivial logical step at a time.
- Interpret the key equation or quantity near where it appears.
- Recover a familiar limiting or baseline case when introducing a generalized construction.
- Keep structural identities separate from dynamical consequences.
- Keep existence, compatibility, optimization, and stability statements distinct.
- Do not turn numerical evidence into a theorem-like statement.
- Prefer a transparent derivation over theorem-proof packaging unless the formal structure adds value.

## Local notation first

Before adding notation, inspect the nearest equations and macros in the current manuscript.

Reuse the established:
- symbol;
- index convention;
- bold/plain convention;
- hat/bar/tilde convention;
- calligraphic/roman convention;
- eigenvalue ordering;
- layer/path/mode notation.

Do not replace a functioning project notation with a globally preferred alternative.

The same quantity should use the same notation in:
- manuscript equations;
- prose;
- figure labels and legends;
- code comments and variable names where practical.

## Notation style

The source papers favor light notation: typography should encode a real distinction, not decoration.

### Scalars and parameters

Use ordinary italic Latin or Greek symbols for scalar variables and control parameters:

[
alpha,quad ho,quad 	au,quad p,quad q.
]

Prefer short meaningful subscripts for states, barriers, thresholds, or modes:

[
	heta_a,quad 	heta_b,quad ho_ell,quad 	au_c,quad lambda_2.
]

Do not create decorated variants when a simple subscript is sufficient.

### Superscripts

Superscripts often encode structural family, path order, layer, or model:

[
p^{(d)},qquad L^{(2)},qquad x_i^{[alpha]},qquad
lambda_2^{mathcal M}.
]

Use them for a genuine structural distinction, not as prose packed into mathematics.

### Hats, bars, and other adornments

Use an adornment only when it has a stable semantic role.

Observed patterns:
- a hat can denote an extended or modified object relative to a baseline, e.g. (hat L), (hatlambda_2);
- a bar can denote an averaged/effective quantity when the project already uses that convention;
- angle brackets can denote averages, e.g. (langle zetaangle).

Do not stack hats, bars, tildes, and superscripts unless the distinction is necessary.

### Vectors, matrices, and operators

There is no single boldface convention across all papers. Preserve the local manuscript family.

When creating notation from scratch:
- keep scalars plain italic;
- distinguish vectors/matrices typographically only when that distinction improves the argument;
- use one vector convention consistently (`\mathbf{}` or `\bm{}`), not both;
- use calligraphic letters for global structures/energies/operators only when that semantic role is useful;
- use roman type for named operators or textual mathematical labels.

Examples:

[
mathcal E,qquad
mathcal L,qquad
mathrm{Re},lambda,qquad
mathrm{Hol}_alpha(C),qquad
mathrm{grad},qquad
mathrm{div}.
]

Do not bold every object merely because it is technically a vector.

## Derivation pattern

For a new result, prefer this order when applicable:

1. define the object;
2. state the assumptions;
3. recover the familiar case;
4. write the governing relation;
5. reduce or transform it;
6. isolate the quantity that controls the result;
7. state the condition, identity, bound, or threshold;
8. interpret it;
9. check a limiting case or numerical consequence.

Skip steps that add no information.

## Mathematical status

Keep the status of a statement visible.

Exact:
- "is equal to";
- "if and only if";
- "follows from";
- "the kernel is nontrivial precisely when".

Conditional:
- "under the assumption that...";
- "for this graph class...";
- "within this regime...".

Approximate:
- "to first order";
- "in the large-X limit";
- "within the mean-field approximation";
- "numerically, we find...".

Interpretive:
- "can be interpreted as";
- "suggests";
- "provides a measure of".

Do not move silently between these levels.

## Assumptions and scope

Attach conditions to the result they constrain.

State relevant assumptions such as:
- connectivity;
- symmetry or directedness;
- normalization;
- graph class;
- parameter range;
- perturbative regime;
- independence or mean-field assumptions.

Do not let a result proved for a restricted class read as a general statement later.

## Generalizations

When an object extends a standard construction, show the reduction explicitly.

Examples:
- zero phase lag -> ordinary incidence or Laplacian;
- no cycles -> no holonomy obstruction;
- vanishing relaxation -> the appropriate diffusive limit;
- zero perturbation -> the baseline operator.

Do not write only that something "generalizes" a familiar object.

## LaTeX source style

- Use `$...# Mathematical reasoning and notation

Use this module for derivations, notation, analytical arguments, approximations, spectral reasoning, and mathematical interpretation.

## Core rules

- Start from the concrete mathematical, physical, network, or geometric object.
- Introduce notation only when it carries information needed later.
- Use the smallest useful set of symbols.
- Do not introduce an extra parameter unless it carries independent information.
- Keep each symbol's meaning stable.
- State assumptions before they are used.
- Move through one nontrivial logical step at a time.
- Interpret the key equation or quantity near where it appears.
- Recover a familiar limiting or baseline case when introducing a generalized construction.
- Keep structural identities separate from dynamical consequences.
- Keep existence, compatibility, optimization, and stability statements distinct.
- Do not turn numerical evidence into a theorem-like statement.
- Prefer a transparent derivation over theorem-proof packaging unless the formal structure adds value.

## Local notation first

Before adding notation, inspect the nearest equations and macros in the current manuscript.

Reuse the established:
- symbol;
- index convention;
- bold/plain convention;
- hat/bar/tilde convention;
- calligraphic/roman convention;
- eigenvalue ordering;
- layer/path/mode notation.

Do not replace a functioning project notation with a globally preferred alternative.

The same quantity should use the same notation in:
- manuscript equations;
- prose;
- figure labels and legends;
- code comments and variable names where practical.

## Notation style

The source papers favor light notation: typography should encode a real distinction, not decoration.

### Scalars and parameters

Use ordinary italic Latin or Greek symbols for scalar variables and control parameters:

[
alpha,quad ho,quad 	au,quad p,quad q.
]

Prefer short meaningful subscripts for states, barriers, thresholds, or modes:

[
	heta_a,quad 	heta_b,quad ho_ell,quad 	au_c,quad lambda_2.
]

Do not create decorated variants when a simple subscript is sufficient.

### Superscripts

Superscripts often encode structural family, path order, layer, or model:

[
p^{(d)},qquad L^{(2)},qquad x_i^{[alpha]},qquad
lambda_2^{mathcal M}.
]

Use them for a genuine structural distinction, not as prose packed into mathematics.

### Hats, bars, and other adornments

Use an adornment only when it has a stable semantic role.

Observed patterns:
- a hat can denote an extended or modified object relative to a baseline, e.g. (hat L), (hatlambda_2);
- a bar can denote an averaged/effective quantity when the project already uses that convention;
- angle brackets can denote averages, e.g. (langle zetaangle).

Do not stack hats, bars, tildes, and superscripts unless the distinction is necessary.

### Vectors, matrices, and operators

There is no single boldface convention across all papers. Preserve the local manuscript family.

When creating notation from scratch:
- keep scalars plain italic;
- distinguish vectors/matrices typographically only when that distinction improves the argument;
- use one vector convention consistently (`\mathbf{}` or `\bm{}`), not both;
- use calligraphic letters for global structures/energies/operators only when that semantic role is useful;
- use roman type for named operators or textual mathematical labels.

Examples:

[
mathcal E,qquad
mathcal L,qquad
mathrm{Re},lambda,qquad
mathrm{Hol}_alpha(C),qquad
mathrm{grad},qquad
mathrm{div}.
]

Do not bold every object merely because it is technically a vector.

## Derivation pattern

For a new result, prefer this order when applicable:

1. define the object;
2. state the assumptions;
3. recover the familiar case;
4. write the governing relation;
5. reduce or transform it;
6. isolate the quantity that controls the result;
7. state the condition, identity, bound, or threshold;
8. interpret it;
9. check a limiting case or numerical consequence.

Skip steps that add no information.

## Mathematical status

Keep the status of a statement visible.

Exact:
- "is equal to";
- "if and only if";
- "follows from";
- "the kernel is nontrivial precisely when".

Conditional:
- "under the assumption that...";
- "for this graph class...";
- "within this regime...".

Approximate:
- "to first order";
- "in the large-X limit";
- "within the mean-field approximation";
- "numerically, we find...".

Interpretive:
- "can be interpreted as";
- "suggests";
- "provides a measure of".

Do not move silently between these levels.

## Assumptions and scope

Attach conditions to the result they constrain.

State relevant assumptions such as:
- connectivity;
- symmetry or directedness;
- normalization;
- graph class;
- parameter range;
- perturbative regime;
- independence or mean-field assumptions.

Do not let a result proved for a restricted class read as a general statement later.

## Generalizations

When an object extends a standard construction, show the reduction explicitly.

Examples:
- zero phase lag -> ordinary incidence or Laplacian;
- no cycles -> no holonomy obstruction;
- vanishing relaxation -> the appropriate diffusive limit;
- zero perturbation -> the baseline operator.

Do not write only that something "generalizes" a familiar object.

 for inline mathematics. Do not use `\\(...\\)`.
- Use `$...$` for unnumbered display mathematics in notes or Markdown. Do not use `\\[...\\]`. In LaTeX manuscripts, keep `equation`, `align`, or related environments when numbering, labels, or alignment are needed.
- Make the equation source easy to read. Put spaces around binary and relation operators, and separate multiplicative factors instead of compressing them together.
- Prefer `d_q = 2 g ( 1 - \\cos q )` over `d_q=2g(1-\\cos q)`.
- Keep standard function arguments compact, e.g. `\\tanh(a_*)`, while spacing the surrounding factors, e.g. `2 g \\tanh(a_*) \\sin q`.
- Apply this spacing consistently to inline and displayed mathematics.

## Equation style

Treat displayed equations as part of the sentence.

- Introduce the equation in prose.
- Punctuate the display when the sentence continues or ends there.
- Define new symbols immediately after first use.
- Prefer one conceptual step per display when practical.
- Use aligned/split environments only when they make a real derivation easier to read.
- Avoid naming an equation or quantity that appears only once.
- Avoid dense displays that mix definition, approximation, and conclusion.

When a result can be expressed as a clean identity, equivalence, threshold, or bound, make that expression visually central.

## Spectral arguments

When using eigenvalues or modes:
- state the operator or matrix assumptions relevant to the spectrum;
- define the eigenvalue ordering;
- identify which mode controls the quantity of interest;
- handle degeneracy when it matters;
- distinguish a local branch from a global variational quantity.

Do not infer monotonicity of an eigenvalue or mode merely from monotonicity of an underlying parameter.

## Approximations

For an approximation, make visible:
- the small or large parameter;
- the baseline object;
- the order retained;
- the resulting approximation;
- the expected regime of validity.

Keep analytical prediction and numerical validation distinct.

## Sanity checks

Use the checks relevant to the derivation:
- zero coupling, lag, or perturbation;
- large or small parameter limit;
- tree or trivial graph;
- dimensions or units;
- signs;
- transpose or adjoint conventions;
- normalization;
- local versus global extremum;
- geometric or spectral threshold versus dynamical stability threshold.

A derivation is not complete because the algebra is internally consistent; it should also survive the relevant structural checks.
