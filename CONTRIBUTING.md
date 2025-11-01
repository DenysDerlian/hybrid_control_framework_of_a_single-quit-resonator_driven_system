# Contributing

Thanks for your interest in contributing! This project is a research codebase focused on quantum control optimization and benchmarking. We welcome improvements, bug fixes, documentation updates, and reproducible experiments.

## Getting started

1. Fork the repo and create a feature branch:
   - Branch naming: `feat/<topic>`, `fix/<issue>`, `docs/<section>`
2. Set up a Python environment (3.10+ recommended) and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If you plan to use NN-assisted components, ensure TensorFlow/GPU is properly installed; otherwise, you can comment it out in `requirements.txt`.

## Development workflow

- Keep changes small and focused; split unrelated changes into separate PRs.
- For public functions in `src/`, prefer docstrings and type hints where possible.
- If you change public behavior, add or update a minimal example in README or the notebooks.
- Run notebooks in order (1→5) to ensure nothing breaks; capture any regressions with a short note.

## Code style

- Python: PEP 8/PEP 257 style where practical
- Use descriptive variable names; keep functions short and modular
- Avoid hardcoding paths; use the existing structure in `data/` and `results/`

## Adding data and results

- Small curated payloads may live under `data/` (kept in Git)
- Large or run-specific payloads should go under `data/payloads/` (ignored)
- Results, figures, and analysis go to `results/` (ignored)

## Commit messages

- Use imperative mood: "Add X", "Fix Y", "Refactor Z"
- Reference issues when appropriate (e.g., `Fixes #12`)

## Opening a pull request

- Ensure your branch is rebased on the latest `main`
- Provide a concise description of what and why
- Include screenshots/plots for visual changes when helpful
- If your change affects the notebooks, point to the relevant cell(s) and outputs

## Reporting issues

- Use GitHub Issues and provide:
  - Clear steps to reproduce
  - Expected vs actual behavior
  - Error messages, stack traces, or screenshots
  - Platform and package versions (e.g., `pip freeze`)

## License

By contributing, you agree that your contributions will be licensed under the MIT License included in this repository.
