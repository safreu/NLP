# Quick Start

[![CI](https://github.com/safreu/NLP/actions/workflows/workflow.yml/badge.svg?branch=dev)](https://github.com/safreu/NLP/actions/workflows/workflow.yml)
[![Tests](https://img.shields.io/badge/tests-GitHub%20Actions-blue?logo=github)](https://github.com/safreu/NLP/actions/workflows/workflow.yml)
[![Coverage](https://img.shields.io/badge/coverage-see%20Actions%20summary-blueviolet?logo=github)](https://github.com/safreu/NLP/actions/workflows/workflow.yml)

```bash
git clone git@gitlab.uni-ulm.de:slb51/nlp.git
cd nlp
uv sync --dev
uv run nox
```

---

# Project Structure
## WIP

- docs -> Documentation regarding this project
- src -> Source directory containing the code
- tests -> Directory containing the tests

## Local outputs and generated files

Experiment outputs are local artifacts and are ignored by Git by default:

- `runs/` stores run-specific predictions, scores, metadata, and model outputs.
- `results/` stores ad-hoc evaluation outputs.
- `models/` stores local model checkpoints.
- `data/**/*.pkl` stores generated dataset caches.

Keep reproducible source data, code, and documentation in Git. Do not commit generated run outputs unless they are intentionally curated examples for the report.
---

# Development Setup

Dependencies are split into two groups:

- **Runtime dependencies**: required to run the project (e.g. `transformers`, `datasets`)
- **Development dependencies**: required for development (e.g. `pytest`, `jupyter`, `ruff`, `mypy`)

## Install runtime dependencies

```bash
uv sync
```

## Install development dependencies

```bash
uv sync --dev
```

## Add a new runtime dependency

```bash
uv add package_name
```

## Add a new development dependency

```bash
uv add --dev package_name
```

---

# Commands

## Run tests

```bash
uv run nox -s tests
```

## Run the linter

```bash
uv run nox -s lint
```

## Run the type checker

```bash
uv run nox -s typecheck
```

## Run all checks (recommended)

```bash
uv run nox
```

---

# CI and Reports

- CI runs on every push and pull request.
- The Test job generates a summary with total tests, failures, errors, skipped, and line coverage.
- You can view these directly in the GitHub Actions run summary without downloading artifacts.

Where to see the numbers:

- Open the latest CI run in GitHub Actions.
- Click the Test job.
- The run summary shows test counts and line coverage percentage.

Coverage and test reports are still uploaded as artifacts for deeper inspection when needed.

---

# Running the Project

## Run a file directly

```bash
uv run python path/to/file.py
```

## Define and use an entry point (recommended)

Project structure:

```
src/nlp/
  __init__.py
  main.py
```

Example `src/nlp/main.py`:

```python
def main() -> None:
    print("setting up")


if __name__ == "__main__":
    main()
```

Define the script in `pyproject.toml`:

```toml
[project.scripts]
src = "main:main"
```

Explanation:

- `src` := command name  
- `main` := refers to `src/main.py`  
- `main` := function that will be executed  

Run the entry point:

```bash
uv run scr
```

---

# Understanding the CI/CD pipeline

- On every push to any branch:
  - the **linter** runs
  - the **tests** run

- On merge requests targeting `main`:
  - the **typechecker** runs in addition

The typechecker is configured in **strict mode**, which means:

- all functions must have type annotations  
- type errors will fail the pipeline  

Will fail:
```python
def main():
    print("setting up")

if __name__ == "__main__":
    main()
```
Will succeed:
```python
def main() -> None:
    print("setting up")


if __name__ == "__main__":
    main()
```


If the typecheck fails, the merge request cannot be merged.

---

## Merges to main

It is recommended to create a merge request describing:

- what changes were made  
- how they were implemented  

## Commit messages

It is recommended to use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
e.g.:

```bash
git commit -m "feat(transformer): implement new transformer library"
```
