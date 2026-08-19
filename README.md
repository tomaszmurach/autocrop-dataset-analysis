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

Dataset audit, conservative pair discovery, and experimental manifest-based
crop reconstruction are implemented. Statistical analysis and optimization are
not implemented.

An isolated content-pairing feasibility experiment is also available. It is
not part of the deterministic filename/stem pairing pipeline and is not
production-proven.

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

## Experimental exact-frame content provenance

The experimental command tests whether a manually cropped, resized, and
recompressed image has uniquely strong evidence of derivation from one exact
source frame. It applies EXIF normalization in memory, SIFT feature extraction,
exact descriptor matching, similarity-transform RANSAC verification, and
aligned luminance/gradient comparison.

Originals retain at most 3,000 SIFT features during a run; their decoded
full-resolution pixels are released after feature extraction. An original is
opened read-only again only when plausible geometry requires aligned
photometric verification. Crops remain cached because the feasibility sample
contains few, comparatively small crop images.

```powershell
python -m autocrop_analysis.content_match_cli `
  --originals C:\path\to\candidate-originals `
  --cropped C:\path\to\manual-crops `
  --output C:\private-output\content-results.private.json
```

All paths are explicit. Input roots must be distinct and non-nested. The
private JSON parent must already exist, the result must be outside both input
roots, and existing output is never replaced. Source images are decoded
read-only and are never normalized or rewritten. Normal console output contains
aggregate counts only.

Content-provenance schema `1.1` records each crop's and candidate original's
EXIF-normalized display dimensions for downstream geometry consumers. Crop
reconstruction consumes this private manifest without reopening either image
tree.

The external decisions mean:

- `MATCHED` / `UNIQUE_STRONG_PROVENANCE`: one candidate passes every
  descriptor, geometry, coverage, transform, photometric, and runner-up
  separation requirement;
- `AMBIGUOUS` / `AMBIGUOUS_PROVENANCE`: multiple candidates remain credible or
  observationally indistinguishable inside the crop;
- `NO_MATCH` / `NO_VALID_PROVENANCE`: no supplied source has adequate,
  internally consistent provenance evidence.

Unique provenance is permitted only when every supplied original image
candidate was evaluable. An audit-unavailable, feature-extraction-unavailable,
or lazy photometric-decode-unavailable original makes the candidate set
incomplete, forcing every crop to `AMBIGUOUS` with the diagnostic
`INCOMPLETE_CANDIDATE_SET`. Aggregate completeness counts are stored in the
private result and printed without filenames.

Real private samples are treated as unlabeled provenance discovery. A
`MATCHED` result is an experimental evidence-based prediction, not ground
truth, and the command does not report accuracy. Correctness and provisional
thresholds are exercised using deterministic labeled synthetic tests. The
current command exhaustively compares small candidate sets; retrieval remains
separate from its provenance decisions.

## Experimental content candidate retrieval

Candidate retrieval builds a persistent private index of spatially balanced,
compact SIFT descriptors and uses exact pooled BF-L2 neighbor voting to return
ranked source-original shortlists. It performs no mutual matching, geometry,
RANSAC, image alignment, or photometric verification.

Build the original-corpus index once:

```powershell
python -m autocrop_analysis.candidate_retrieval_cli build `
  --originals C:\path\to\candidate-originals `
  --output C:\private-output\candidate-index.private.json
```

The JSON index manifest is published beside a raw little-endian float32
`*.descriptors.private.f32` matrix. Both files contain private client-derived
data and must remain outside Git. The manifest binds every semantic original
reference to its encoded-file SHA-256, display dimensions, representation
status, and contiguous descriptor range. The descriptor file is size- and
SHA-256-validated before use.

Query the immutable index with explicit cropped-image input:

```powershell
python -m autocrop_analysis.candidate_retrieval_cli query `
  --index C:\private-output\candidate-index.private.json `
  --cropped C:\path\to\manual-crops `
  --output C:\private-output\retrieval.private.json `
  --k 50
```

Retrieval output contains vote counts and L2-distance diagnostics, not
`MATCHED`, `AMBIGUOUS`, or `NO_MATCH` provenance decisions. Retrieval index
completeness and query completeness are separate from the exact verifier's
`candidate_set_complete` contract. The exhaustive exact verifier remains
unchanged and no shortlist integration is implemented.

The configurable synthetic benchmark uses known constructed sources and may
therefore report Recall@K:

```powershell
python -m autocrop_analysis.candidate_retrieval_benchmark `
  --corpus-size 32 `
  --queries 20 `
  --k-values 1,5,10,20,50,100
```

### Exact-retrieval baseline profiling

Candidate-retrieval profiling is opt-in research instrumentation for measuring
the existing exact BF-L2 reference backend. It does not change retrieval or
provenance semantics, and profiling data is never added to the persistent index
or retrieval-result schemas. Supply a separate no-clobber private JSON output
when building or querying:

```powershell
python -m autocrop_analysis.candidate_retrieval_cli build `
  --originals C:\path\to\candidate-originals `
  --output C:\private-output\candidate-index.private.json `
  --profile-output C:\private-output\candidate-index-profile.private.json

python -m autocrop_analysis.candidate_retrieval_cli query `
  --index C:\private-output\candidate-index.private.json `
  --cropped C:\path\to\manual-crops `
  --output C:\private-output\retrieval.private.json `
  --profile-output C:\private-output\retrieval-profile.private.json `
  --k 50
```

Build profiling separates corpus audit, the first encoded-file hash, combined
feature extraction, the second stability hash, compact selection, descriptor
writing with incremental hashing, manifest/corpus-identity construction, and
publication. Feature extraction intentionally remains one stage containing
decode, EXIF normalization, grayscale conversion, SIFT, and stable feature
ordering because the baseline extractor performs those operations together.

Query profiling separates index-manifest loading/validation, the required
descriptor-binary integrity hash, memmap creation, crop audit, combined query
feature extraction, compact selection, exact BF search, Python vote
aggregation/ranking, shortlist construction, result-manifest construction, and
output publication. It labels the first query and subsequent queries while one
loaded index is shared by the batch. The descriptor work-unit count is selected
query rows multiplied by indexed rows; it is a normalization value, not a CPU
instruction count.

Timings use `time.perf_counter_ns()`. Filesystem page-cache state is uncontrolled:
a normal CLI invocation labels its load `process_fresh_load`, not a guaranteed
cold-disk measurement. The in-process synthetic benchmark labels its load
`same_process_after_build`. On Windows, memory snapshots report current and lifetime-peak
process working set through PSAPI. Other platforms report standard-library
process metrics where available and use explicit null values otherwise; Python
heap usage is never mislabeled as process RSS.

The synthetic benchmark includes the same profiling report in its console JSON.
Tiny or bounded runs are measurements of those runs only and must not be treated
as full-corpus performance predictions. Retrieval profiling measures candidate
retrieval cost; it says nothing about provenance correctness.

Large progression benchmarks are operator-directed and are not part of the
normal unit suite. Future real bounded checks must be described only as
consistency with existing provenance predictions, not retrieval accuracy or
ground truth.

## Experimental crop reconstruction

Crop reconstruction converts a content-provenance schema `1.1` manifest into a
separate private geometry manifest. It does not discover images, reopen source
or crop files, or rerun content matching.

```powershell
python -m autocrop_analysis.crop_reconstruction_cli `
  --provenance C:\private-output\content-results.private.json `
  --output C:\private-output\crop-reconstruction.private.json
```

Both paths are required and must end with `.private.json`. The input must exist,
the output parent must already exist, and an existing output is never replaced.
Normal console output contains aggregate counts only.

Reconstruction is attempted only for a complete candidate set and a
`MATCHED` / `UNIQUE_STRONG_PROVENANCE` crop whose rank-1 evidence is complete,
strong, geometrically valid, and plausible. `AMBIGUOUS` and `NO_MATCH` crops are
preserved as `NOT_RECONSTRUCTED` and never receive selected geometry. A matched
crop can also remain `NOT_RECONSTRUCTED` when its geometry is degenerate,
inconsistent, more than the provisional reconstruction-only 1° from axis
alignment, or outside the original display extent.

Authoritative rectangles are floating-point `left`, `top`, `right`, `bottom`
coordinates in `EXIF_NORMALIZED_DISPLAY_PIXEL_BOUNDARIES`, with half-open
`[left, right) × [top, bottom)` semantics. They are derived from the base RANSAC
projected corners; photometric refinement shifts remain scoring-only. The
result is experimental provenance-derived geometry, not independently verified
ground truth.

## Development

Python 3.13 or newer is required.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m unittest discover -s tests -v
```

Runtime dependencies are Pillow, NumPy, and headless OpenCV. Tests use Python's
standard-library `unittest` framework and create synthetic images only in
temporary directories.
