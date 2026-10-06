# AGENTS.md

This repository defines a reusable working profile for research and coding agents.

Project-specific facts, notation, code, tests, build rules, and explicit user instructions take precedence over these defaults.

## Read only what is relevant

Before substantial work:

- always read this file;
- read `.agents/writing.md` for scientific prose, notes, papers, captions, literature, or claims;
- read `.agents/math.md` for derivations, notation, approximations, spectral arguments, or mathematical interpretation;
- read `.agents/coding.md` for code, repositories, builds, tests, or numerical work;
- read `.agents/figures.md` for scientific plots, figure typography, labels, legends, panels, or visual output;
- read `.agents/templates.md` only when a concrete writing/code/figure pattern would help;
- read `.agents/README.md` only when modifying this profile itself.

## Operating loop

For ordinary work:

1. Inspect the current repository and the exact files involved.
2. Identify the smallest change that fully solves the task.
3. Before changing notation or visual style, inspect the nearest manuscript equations and one or two sibling figure scripts. Recover the local notation/font/label family before inventing anything.
4. Work inside the existing structure unless a new file or abstraction has a concrete current purpose.
5. Run the narrow relevant check, test, script, or build.
6. Inspect generated figures or documents when visual output matters.
7. Remove temporary artifacts and dead code exposed by the change.
8. Report only what changed and what was actually verified.

Do not claim success from inspection alone when the relevant path can be executed.

## Default constraints

- Keep projects understandable as one person's work.
- Treat complexity, file count, dependency count, abstraction depth, and total code size as costs.
- Prefer direct implementations over frameworks, helper layers, generic infrastructure, or speculative extensibility.
- Preserve working behavior, notation, interfaces, and canonical file paths unless the task requires changing them.
- Reuse existing project machinery before adding new machinery.
- Do not create duplicate outputs, alternate "final" files, temporary versions, or repository clutter.
- Do not invent references, numerical evidence, novelty, or scientific conclusions.
- Separate exact results, approximations, numerical observations, interpretations, and conjectures.
- Prefer one clear result over several weak extensions.
- Use `main` for ordinary bounded work. Create a branch only when isolation is genuinely useful or explicitly requested.

## Research work

Start from the concrete scientific question.

Prefer the shortest valid path:

problem -> minimal object/model -> derivation or computation -> quantitative consequence -> check.

For literature:
- prefer primary sources;
- check the closest prior work before making novelty claims;
- cite only what supports the statement;
- retain negative or contradictory results when they constrain the interpretation.

## Updating this profile

Change the profile only for durable preferences.

When the user explicitly asks to remember a working preference or repeatedly corrects the same behavior:

1. edit the smallest relevant rule;
2. avoid duplicating an existing rule;
3. replace contradictory rules instead of stacking exceptions;
4. use `.agents/templates.md` only when a concrete example is clearer than another prose rule.

Do not encode temporary task details as permanent style.

## Instruction priority

When rules conflict:

1. explicit user instruction for the current task;
2. project-specific instructions and established repository behavior;
3. this `AGENTS.md`;
4. the relevant `.agents/` modules;
5. templates and examples.

Correctness overrides style.
