# C5 offline experiment skeleton

No acquisition or image classification occurs here. A future NAD importer and
independent Census/land-use joins supply candidate records. Required fields:

```json
{"id":"unique-survey-id","site_id":"stable-site-id","group_id":"source-neighborhood-id",
 "group_basis":"source/version/method","source":"NAD release reference",
 "lat":39.0,"lon":-77.0,
 "strata":{"region":null,"settlement":null,"use":null},"strata_sources":{}}
```

Nonmissing strata require independent source references in `strata_sources`.
Missing labels remain a missing stratum; NAD alone is not a use/urbanity label.
The development input uses the same record format and must enumerate all six
existing development cases. Candidates sharing site IDs, neighborhood IDs or a
connected chain of locations within 1 km are grouped before sampling. Entire
groups touching development cases are excluded. Repeated surveys must share site
IDs. Inputs are limited to 10,000 rows for this quadratic feasibility prototype;
large candidate frames need a spatial-index importer first.

`python pipeline/validation/harness.py freeze candidates.json development.json frozen.json`

Selection cycles available strata, targets 60 **groups**, and assigns whole groups
to tuning/calibration/evaluation. Address counts may differ and crossed strata
may be unbalanced. Input, development and manifest hashes are recorded. Outputs
use exclusive creation and refuse replacement. Keep source manifests/versioned
artifacts outside tuning edits; hashes detect change but are not access controls.

`python pipeline/validation/harness.py summarize frozen.json results.json evaluation metrics.json`

Results are `[{"id":"...","metrics":{"roof_median_abs_ft":0.4}}]`. Every absent
case/metric stays missing/incomplete. Results outside the requested split and
modified manifests are rejected. This is availability accounting, not the final
scoring implementation: entrance association/rank, storey/material ground truth,
block withholding, canopy thinning, confidence intervals and classifier evaluation
remain to be implemented with independently labelled data. It provides no calibrated
probabilities. The six development cases and manual R1 references cannot become
the calibration/evaluation set. Do not inspect evaluation results to choose methods
or thresholds. No images, occupant/owner data or network tokens are used.

`python -m unittest discover -s pipeline/validation -p 'test_*.py' -v`
