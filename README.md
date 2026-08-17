# AutoCrop Dataset Analysis

This repository is a small, developer-side utility project for inspecting and
analyzing photo datasets used to improve AutoCrop behavior. It is intentionally
separate from the production AutoCrop application and must not depend on or
modify that application's repository.

Private client images and datasets must remain outside this repository and
outside Git. Future commands will accept operator-supplied dataset paths
explicitly; no dataset location is assumed or created here. Generated manifests
and reports may expose filenames, absolute paths, or image metadata, so they
must also remain untracked.

## Intended pipeline

```text
dataset audit
→ pair discovery
→ ground-truth crop reconstruction
→ statistical analysis
→ parameter optimization
→ frozen evaluation
```

Only the project foundation exists today. Dataset scanning, pair matching,
crop reconstruction, statistical analysis, and optimization are not yet
implemented.

## Development

Python 3.13 or newer is required.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m unittest discover -s tests -v
```

Pillow is the sole runtime dependency, reserved for the upcoming image metadata
audit. Tests currently use Python's standard-library `unittest` framework.

