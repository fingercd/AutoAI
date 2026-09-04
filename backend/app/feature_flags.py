"""Internal product gates that are intentionally not user configurable."""

# TEMPORARILY_HIDDEN: do not re-enable without an explicit product requirement
# and synchronized backend/frontend contract tests.
#
# This is deliberately a source constant rather than an environment variable or
# request option.  New training runs must not spend compute on explainability or
# model-feature visualisation until the product contract is explicitly restored.
EXPLAINABILITY_ENABLED = False
