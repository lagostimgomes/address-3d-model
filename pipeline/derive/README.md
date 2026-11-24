# Imagery-derived observations

Optional inputs, one site folder at a time, written to `<site>/observations/`. The pipeline uses them when present.

- `ground.json`: ground cover (asphalt, concrete, lawn, bed) on a 1 ft grid, from a georeferenced aerial mosaic.
  `python pipeline/derive/aerial_ground.py <site> <mosaic.png> <mosaic.pgw> <site>/observations/ground.json`.
  Fixed HSV rules. Building footprints are masked, compact grey blobs not joined to the street network are dropped,
  and thick light-grey blobs (roofs) are dropped.
- `looks.json`: facade observations per building id (front/sides material and colour, shutters, door, garage, front
  gable, entry, plantings), read from street-level stills into a fixed schema. Status `observed`, with source and image
  date. So far these were filled by a vision-model reading of the stills (Claude). The schema is the
  contract; a scripted reader is the next step.

Imagery pixels never enter the repo, scene.json or a model. Only these derived records do. Google imagery is used under
the user's decision for personal, non-commercial research (see collaboration notes); models built with it are not
published.
