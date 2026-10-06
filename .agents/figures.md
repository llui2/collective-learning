# Scientific figures

Use this module for manuscript figures, scientific plotting, figure typography, labels, legends, panels, and visual validation.

Before using these defaults, inspect one or two sibling figures from the same paper. Project-local figure style takes precedence.

## Recover the local figure family first

Before creating or restyling a manuscript figure:

1. inspect one or two sibling figure scripts from the same paper;
2. inspect the manuscript class/template and, if available, the rendered neighboring figures;
3. recover the local font family, math font, base size, panel-label convention, line widths, palette, output format, and label notation;
4. reuse that family unless the user explicitly asks for a new style.

Do not impose one universal font across all projects.

The corpus contains both:
- serif text + Computer Modern-style math;
- Helvetica/Arial/DejaVu Sans-style text + matching sans math.

The stable preference is internal consistency and compatibility with the manuscript.

## Typography

Use one coherent typography scale per figure family.

Typical hierarchy:
- tick labels and legend: base size;
- axis labels: slightly larger than ticks;
- panel labels: slightly larger and bold;
- annotations: around base size unless they are central.

Do not set unrelated font sizes independently throughout the script.

For serif figures, a Times-like serif with Computer Modern math is an established pattern.

For sans-serif figures, Helvetica/Arial/DejaVu Sans with sans-compatible math is an established pattern.

Avoid mixing a serif text family with unrelated sans math, or vice versa, unless the existing paper already does so deliberately.

Panel letters are commonly bold sans-serif even when the rest of the figure is serif.

## Axis labels

Prefer the manuscript's actual mathematical notation.

For a canonical mathematical quantity, use the symbol directly:

`r"$\rho$"`, `r"$\theta$"`, `r"$U(\theta)$"`, `r"$\lambda_1$"`.

For a quantity whose meaning is not obvious from the symbol alone, use a short descriptive phrase plus the symbol:

`r"cycle current $|q|$"`

`r"modal decay rate $-\mathrm{Re}\,\lambda$"`

For physical/empirical quantities, put units in parentheses:

`r"frequency $f$ (Hz)"`

`r"relaxation time $\tau$ (s)"`.

Use sentence case for prose labels. Preserve acronyms and proper names.

Do not write a long explanatory sentence as an axis label.

Use raw strings for LaTeX-like labels when practical.

## Tick labels

Keep ticks sparse and meaningful.

- Use exact symbolic values when the parameter has natural mathematical landmarks, e.g. (0,pi/4,pi/2,pi), rather than decimal approximations.
- Use simple decimal labels when exact notation would add clutter.
- On log axes, choose a small readable set of fixed ticks when the default formatter is noisy.
- Do not add minor ticks merely because Matplotlib can.

## Spines and grids

The current preferred style is open and light.

- no grid by default;
- hide top and right spines when compatible with the plot;
- keep the remaining axes thin;
- avoid boxed/dashboard-like frames.

Preserve a closed-axis convention if the existing figure family clearly uses one.

## Titles and in-plot text

Avoid a large generic plot title when the caption already provides the context.

Use titles only when they distinguish scientifically meaningful panels/cases.

For a single parameter value, prefer a compact edge annotation such as

`r"$\rho=1$"`

over a decorative title.

Keep annotations sparse and mathematical when possible.

## Legends

Legends should be short and semantic.

- use `frameon=False`;
- do not add a legend title unless it carries information;
- keep entries to a symbol or short phrase;
- place the legend where it does not cover the relevant data;
- move a shared legend above/below a panel group when this reduces clutter;
- construct explicit `Line2D`/`Patch` handles when the legend describes regions, manifolds, thresholds, or other semantics not represented by one plotted line.

Examples of suitable entries:
- `r"$\rho_\ell$"`;
- `r"stable $W^s$"`;
- `"conservative boundary"`.

## Panel labels

Use simple lowercase panel letters.

Default placement is near the upper-left edge of the panel, often slightly outside the axes.

Use bold sans-serif panel labels for visibility.

Match the current paper's convention for `a` versus `(a)`; do not mix both within one figure set.

## Lines, markers, and color

Keep line art thin enough for manuscript figures.

Typical line widths are around 1--2 points; use heavier lines only for the main object or boundary.

Keep markers small unless individual observations are the point.

Use black/gray for:
- baselines;
- theory/reference curves;
- threshold lines;
- neutral structural elements.

Use color to distinguish scientific categories/conditions. Keep the palette small and stable across figures.

Prefer restrained colors in new work. Do not revive older bright/neon palettes merely because they appear in historical scripts.

Use line style as a semantic channel when useful:
- solid/dashed/dotted for theory, approximation, threshold, or alternative condition;
- do not vary color, marker, and linestyle simultaneously unless the figure needs all three.

## Layout and size

Design at the intended manuscript size.

- single-column figures should be compact;
- multi-panel figures should use width for the panels rather than oversized text;
- preserve enough margin for labels without large dead space;
- use `tight_layout`, constrained layout, or explicit `subplots_adjust` according to what renders best.

Do not choose figure size from a default template without checking the final manuscript width.

## Output

Generate a file rather than opening an interactive window.

- do not use `plt.show()` in reproducible repository scripts;
- prefer `fig.savefig(...)`;
- close figures after saving;
- use `Path` for paths in current code;
- keep console output short: useful computed values and the saved artifact path.

Prefer vector PDF for line art when the manuscript workflow supports it.

Use PNG when the project requires raster output or when the existing build uses PNG; use sufficient DPI for the final physical size.

Do not save PDF + PNG + SVG variants by default. Produce the canonical artifact(s) the project actually uses.

## Plot choice

Choose the visualization that tests the claim directly.

- transition -> show the threshold;
- approximation -> plot theory and simulation together;
- comparison -> show the quantities on directly comparable axes or plot the difference;
- parameter regimes -> phase diagram;
- unfamiliar object -> small schematic before quantitative detail.

Do not plot secondary quantities merely because they are available.

## Validation scale

"Code ran" is not enough.

At calculation level:
- check dimensions, finite values, signs, invariants, and relevant limits.

At figure level:
- inspect the rendered artifact;
- check font consistency;
- check clipping, overlap, whitespace, axis range, labels, legend placement, and manuscript-scale readability;
- verify mathematical symbols match the manuscript notation exactly;
- confirm the claimed trend is visible without reading the source code.

At repository level:
- run the narrow relevant script/test;
- run the canonical build when practical;
- remove temporary artifacts and dead code;
- stop when the requested behavior is correct and clear.

When two implementations are equally correct and readable, choose the shorter one.
