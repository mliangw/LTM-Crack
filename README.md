# LTM-Crack synthetic generator and evaluation code

This package provides the synthetic-mask generator and associated evaluators for *LTM-Crack: A Locate-Then-Measure Workflow for Candidate-Site Crack-Width Measurement on Concrete Bridge Decks*.

## Install

Use Python 3.12 and install the supplied dependency versions:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements-validated.txt
```

The synthetic programs run on a CPU and do not require a trained segmentation model.

## Generate the benchmark

Run from this package directory:

```bash
python generate_synthetic_masks.py --output-dir data/synthetic
```

The default configuration, `configs/balanced160.csv`, generates 160 masks in four nominal-width groups of 40 samples each. The output directory must be empty. The program writes binary masks, design-reference widths and coordinates, and generation settings. Add `--save-overlays` for optional visualizations.

## Co-located local-width comparison

```bash
python evaluate_synthetic.py --syn-dir data/synthetic --output-dir outputs/synthetic_gt --center-mode gt --no-plots
```

This evaluates MCM, OP, OB, ESD, EOB, and EDT at common sites determined from the stored design-reference coordinates, with nearest-skeleton snapping when eligible. The CSV records both the reference and actual measurement coordinates.

The alternative `--center-mode auto` evaluates all estimators at the common EDT-selected candidate. These modes use different sites and should be reported separately. Omit `--no-plots` to generate the comparison figure.

## Automatic maximum-width estimation

```bash
python evaluate_candidate_search.py --syn-dir data/synthetic --output-dir outputs/candidate_search --sample-step 3 --no-plots
```

The program measures MCM at its EDT-selected candidate and compares it with the maximum valid widths from sampled skeleton scans using OP, OB, ESD, and EOB. The EDT baseline reports twice the maximum EDT response over the eligible skeleton domain.

The default sampling step is 3, matching the sampled-scan experiment. Sampling is applied to the skeleton-coordinate list. Use `--sample-step 1` for a separate full traversal. The degree-filtered variants use pixels of degree at most two. Each method retains its specified measurement procedure; this is a comparison of complete automatic estimates.

## Design references

The default scale is 0.01278 mm/pixel. References are the maximum unperturbed design-width profile over the central 60% of 500 centerline samples, recorded before boundary perturbation and integer rasterization. They are not independent measurements of the final mask's maximum width. L1–L4 are nominal generation groups.

See [parameter settings](docs/parameters.md), [evaluation protocols](docs/evaluation_protocols.md), and [input/output formats](docs/reference_formats.md).
