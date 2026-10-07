# Evaluation protocols

## Co-located comparison

All six estimators use one common site in each mask. In `gt` mode, the design-reference coordinate is snapped to the nearest skeleton point if the distance is no greater than the EDT value at that coordinate plus 2 pixels. Otherwise the original coordinate is retained. The CSV records the reference coordinate, actual coordinate, and selected mode.

In `auto` mode, the site is the largest EDT response among degree-at-most-two skeleton pixels, with a full-skeleton fallback if none remain. Results from the two site modes must be reported separately.

## Automatic maximum-width estimates

The candidate-search evaluator applies MCM at its EDT-selected candidate and the OP, OB, ESD, and EOB functions at sampled skeleton coordinates, retaining each estimator's existing measurement procedure. It returns each scan's maximum valid result.

The default step of 3 subsamples the skeleton-coordinate list. A step of 1 is a full traversal. The full-domain and degree-at-most-two variants are reported separately. The comparison concerns complete automatic estimation procedures, including their search and width definitions.

## Reference scope

The stored design reference precedes boundary perturbation and rasterization. Differences from this reference assess recovery of the design width, rather than independent physical measurement uncertainty or proven localization of the final mask's true maximum width.
