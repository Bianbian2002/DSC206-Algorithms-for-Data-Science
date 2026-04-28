# CLAUDE.md — DSC 206: Algorithms for Data Science

## Course Context
**Student:** Zeyu Bian  
**AI Policy:** AI assistance is explicitly allowed and expected for coding, debugging, designing experiments, explaining concepts, and writing.

## How to Help

- Help with all parts of every assignment: theory proofs, Python implementation, experiment design, and writing.
- For theory problems, provide rigorous mathematical proofs with clear step-by-step reasoning. Use LaTeX-style math notation in markdown.
- For coding tasks, write clean Python. Assignments live in `hw<N>/` subdirectories.
- For experiments, help design clear plots and concise discussion connecting empirical results back to the theoretical bounds.

## Repository Structure

```
DSC206-Algorithms-for-Data-Science/
├── CLAUDE.md          ← this file
├── READ.md
├── elegantnote.cls    ← LaTeX class (project root, found via TEXINPUTS)
├── preface.tex        ← shared preamble (project root)
├── hw_template.tex    ← starting template for each homework
└── Assignment/
    ├── hw1/           ← code, plots, and report for Assignment 1
    └── hw<N>/         ← hw<N>.pdf (prompt) + hw<N>.tex (your writeup)
```

## Coding Conventions

- Python for all implementations
- Use `numpy` for numerical work, `matplotlib` for plots
- Keep code in `hw<N>/` with a self-contained script or notebook
- Plots should have labeled axes, titles, and legends

## Homework Workflow

For each new homework `hw<N>`:

1. **Read inputs first.** Open `Assignment/hw<N>/hw<N>.pdf` (the prompt) and `hw_template.tex` (LaTeX template using `elegantnote` class with `homework` and `proof` environments). The shared preamble lives in the project root: `preface.tex` and `elegantnote.cls`.
2. **Write to `Assignment/hw<N>/hw<N>.tex`.** Use `\input{../../preface}` (relative path — `preface.tex` is in the project root, two levels up). Do not duplicate `preface.tex` into the hw directory.
3. **Per-problem structure:** wrap each problem's statement in `\begin{homework}...\end{homework}` and the solution in `\begin{proof}\solutionname ... \end{proof}`. Restate the problem briefly at the top of each `homework` block so the PDF is self-contained.
4. **Proof style:** rigorous but compact. Lead with the key insight, then the formal argument. Use `\MR`, `\MC`, etc. from `preface.tex`. Use `\bm` for vectors and `align*` for multi-line equations.
5. **Compile to verify** before declaring done:
   ```
   cd Assignment/hw<N> && TEXINPUTS=".:../../:" xelatex -interaction=nonstopmode hw<N>.tex
   ```
   The `TEXINPUTS` prefix lets xelatex find `elegantnote.cls` and `preface.tex` in the project root. Check that the PDF builds and inspect the page count.
6. **For coding parts:** put scripts/notebooks in `Assignment/hw<N>/`, save plots as PDFs, and `\includegraphics{...}` them in the report.

## Notes for Claude

- The student understands the material and wants rigorous, correct proofs — don't oversimplify.
- Always explain the key insight behind each proof step, not just the mechanics.
- When writing the AI usage report, be honest and specific about what AI contributed.
- Flag any step where the proof requires a non-trivial lemma so the student can verify they understand it.
