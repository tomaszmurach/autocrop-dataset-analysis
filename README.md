# AutoCrop Dataset Analysis

This repository contains a small, developer-side utility for inspecting photo
datasets used to improve AutoCrop behavior. It is intentionally separate from
the production AutoCrop application and does not depend on or modify that
application's repository.

Private client images and datasets must remain outside this repository and
outside Git. The audit accepts operator-supplied dataset paths explicitly; no
dataset location is assumed or discovered. Its manifest contains private
filenames, paths, and image metadata and must never be committed.

## Intended pipeline

```text
dataset audit
→ pair discovery
→ ground-truth crop reconstruction
→ statistical analysis
→ parameter optimization
→ frozen evaluation
```

Dataset audit and conservative pair discovery are implemented. Crop
reconstruction, statistical analysis, and optimization are not implemented.

## Dataset audit and pair discovery

The command recursively inventories two explicit roots, inspects image metadata
with Pillow, reports malformed or unsupported image candidates, and attempts to
associate each cropped candidate with one original. It never follows symlinks,
junctions, or other reparse points and never writes to either source tree.

```powershell
python -m autocrop_analysis `
  --originals C:\path\to\originals `
  --cropped C:\path\to\manual-crops `
  --output C:\private-output\audit_manifest.private.json
```

All arguments are required. The two input roots must be separate and must not
contain one another. The output parent must already exist, the output must be
outside both roots, and its filename must end with `.private.json`. Existing
output is never overwritten.

Normal console output contains aggregate counts only. One versioned JSON
manifest contains per-file findings and therefore remains private even when its
location is outside this repository.

Pairing uses only two deterministic rules:

1. A complete case-folded filename, including extension, uniquely identifies an
   original.
2. If no complete filename matches, a case-folded stem uniquely identifies an
   original, allowing an extension change.

`MATCHED` means the first applicable rule has one candidate. `AMBIGUOUS` means
that rule has multiple candidates; the tool never falls through to a weaker
rule or chooses a winner. `UNMATCHED` means neither rule has candidates.
Directory similarity, image dimensions, metadata, and image content never
influence pairing.

Pairing includes unreadable and unsupported image candidates so identity does
not depend on installed codecs. A matched pair is reconstruction-ready only
when both files are readable and have the required core dimensions. Pillow may
not decode RAW, HEIC/HEIF, AVIF, or other formats in a particular installation;
such image-like files remain visible as unsupported findings.

## Development

Python 3.13 or newer is required.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m unittest discover -s tests -v
```

Pillow is the sole runtime dependency. Tests use Python's standard-library
`unittest` framework and create synthetic images only in temporary directories.
