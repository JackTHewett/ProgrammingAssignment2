# Perseid meteor pipeline

Processes a night of Perseid frames into a stacked sky, a meteor composite, a
timelapse and a contact sheet of candidate detections.

Built for a Fujifilm X-T30 III session at 13mm f/3.5, shot from a table (so the
framing drifts and gets knocked), with a roofline and railing in frame, heavy
light pollution including a lamp near the edge, and aircraft traffic overhead.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### JPEG-only is fully supported

You do not need the RAF files. The pipeline works end to end from the JPEGs
alone, and that is the only configuration that has been tested — all the
validation below was run on a JPEG-only set.

RAF is used *opportunistically*: for the stacks only, and only when a matching
raw sits next to the JPEG. Detection and the timelapse always use the JPEGs by
design, since speed matters more than shadow latitude there. Pass `--jpeg-only`
to ignore RAFs even when present, and skip installing `rawpy` if you never want
them.

Stacking JPEGs is still worth doing in 16-bit: averaging ~170 frames recovers
roughly 3–4 bits below the 8-bit quantisation step, so the output TIFF holds
real detail that no single JPEG does.

`ffmpeg` on `PATH` is used for the timelapse when present, otherwise the bundled
`imageio-ffmpeg` encoder is used.

## Use

Run from the folder holding the frames, or point at it with `--source`:

```bash
python run.py scan                    # EXIF + grouping report, always cheap
python run.py mask                    # foreground mask + overlay preview
python run.py register                # star alignment                (slow)
python run.py detect                  # meteor candidates             (slow)
python run.py sheet                   # contact sheet + review CSV
python run.py composite               # stacks, gradient, composites  (slow)
python run.py timelapse               # 24fps MP4
python run.py all                     # everything, in order
```

**Start with `scan`.** It reads EXIF from every frame and prints the shooting
session and the grouping, without touching a pixel of processing. Check that the
groups look right before running anything slow.

The three slow stages prompt before processing the whole set. `--yes` skips the
prompt, `--group e1f0` restricts to one group, `--force` ignores the cache.

### The review loop

`detect` classifies each candidate as meteor, aircraft or satellite, but the
classifier only proposes. `sheet` writes `output/contact_sheet_<group>.jpg` with
every candidate labelled by filename, plus `output/candidates_<group>.csv`.

Put `meteor` or `reject` in that CSV's `verdict` column and rerun `composite`.
Hand verdicts always beat the classifier, and blank rows fall back to it.
Verdicts survive a rerun of `sheet`.

Because the stages are cached independently, editing verdicts and rebuilding the
composite does not redo registration — that loop is a few seconds.

### If the mask is wrong

Check `output/previews/mask_overlay_<group>.png`; red is the masked foreground.
If it is wrong, paint a mask by hand (white = foreground) and pass
`--mask-file mymask.png`. This matters more than any other single input: the
mask is what keeps registration off the roofline.

## Output

```
output/
  session_summary.txt            the scan report
  sky_stack_<group>.tif          16-bit, gradient-subtracted, for Lightroom
  meteor_composite_<group>.tif   16-bit, same background + confirmed meteors
  contact_sheet_<group>.jpg      every candidate, labelled
  candidates_<group>.csv         review sheet with the verdict column
  timelapse.mp4                  24fps, unaligned, whole night in shot order
  previews/                      JPEG previews and the mask overlays
  crops/                         per-candidate crops
  .cache/                        intermediate arrays; delete to start over
```

Groups are `e<exposure>f<framing>`. Frames are split by exposure settings and
again wherever the camera moved too far to share one alignment, and each group
stacks separately — so a night with a refocus partway through yields more than
one stack, by design.

## How it works

| Stage | What it does |
| --- | --- |
| `scan` | EXIF per frame; groups by (shutter, ISO) and by framing, using phase correlation on small proxies to find where the camera jumped |
| `mask` | Temporal median over unaligned frames, then marks pixels sitting *below* a smoothed sky model as silhouette |
| `register` | `astroalign` asterism matching on the sky only, with a `DAOStarFinder` fallback; validates every solve |
| `detect` | Subtracts a temporal median of neighbouring frames, runs a probabilistic Hough transform on the residual, scores each trail |
| `sheet` | Crops, labels and tabulates candidates for review |
| `composite` | Sigma-clipped aligned sky stack + unaligned ground stack, blended through the feathered mask; then lighten-blends confirmed meteors |
| `timelapse` | Streams downscaled JPEGs to the encoder in shot order |

### Classifier

Each trail is scored on: whether it continues into an adjacent frame along the
same vector, the fraction of its length that drops to background (strobe
dashes), how many separate maxima it has, its peak-to-mean ratio, its
straightness, and its length and faintness.

A meteor is one frame only, one smooth peak, no gaps. An aircraft is dashed,
multi-peaked, and usually continues across frames. A satellite is long, faint
and also continues.

Note that peak-to-mean ratio is weak evidence on its own and is weighted
accordingly: sharp strobe dashes produce a *higher* peak ratio than a real
meteor, so gap fraction and peak count carry the discrimination.

## Validation

There are no real frames in this repository, so `tests/make_synthetic.py`
generates a night with known ground truth — rotating starfield, table jitter, a
framing break, two exposure settings with real EXIF, a roofline and railing, a
light pollution gradient with a lamp at the edge, three meteors and two
strobing aircraft each spanning three frames.

```bash
python tests/make_synthetic.py /tmp/synth
python run.py all --source /tmp/synth --output /tmp/synth/output --yes
```

Measured on that scene:

| | Result |
| --- | --- |
| Foreground mask | IoU 0.75, purity 75%, **recall 100%** |
| Registration | 40/40 frames, residual 0.13–0.40 px, recovered rotation matches truth |
| Detection | 9 candidates: exactly the 3 meteors and 6 aircraft frames, no false positives |
| Classification | 9/9 correct |
| Gradient removal | residual skyglow 0.134 → 0.032, a 4.2× reduction |

Recall matters far more than purity for the mask: any foreground left showing
through is what breaks registration. The over-claim is mostly the band between
the railing and the roof, which costs a little sky and is otherwise harmless.

**Synthetic frames are not real frames.** They validate that the logic is
correct and the thresholds are not knife-edge, not that the defaults are right
for your sky. Expect to adjust `detection.threshold_sigma` and
`detection.min_trail_length` on real data.

## Colour space

Frames are stacked in **linear light** and flattened in **display space**, which
is a deliberate split:

- Stacking is an average of photons, so it belongs in linear light. It is also
  the only way RAF and JPEG frames can share a stack — `rawpy` is asked for
  linear output while a JPEG carries the sRGB curve, and mixing the two
  unnoticed makes sigma clipping throw out whichever kind is in the minority.
- Flattening goes *after* the transfer curve, which is not where the physics
  would put it. Skyglow adds linearly, so fitting in linear ought to win. It
  measurably does not: the sRGB curve compresses a lamp's dynamic range from
  roughly 18× to 4×, and a low-order polynomial can follow the compressed
  version far better. On the synthetic scene, fitting in display space left
  0.029 residual skyglow against 0.055 for the identical fit in linear.

That second point was checked rather than assumed, and the measurement
contradicted the theory.

## Tuning

All thresholds live in `meteor/config.py`, and the values actually used are
written to `output/config_used.json` on every run.

The ones most likely to need changing:

- `detection.threshold_sigma` (5.0) — raise if the detector returns dozens of
  candidates. A normal Perseid night gives a handful of meteors; dozens means
  it is finding aircraft.
- `detection.min_trail_length` (70 px) — raise to drop short noise streaks.
- `detection.max_line_gap` (30 px) — the bridging distance across strobe
  dashes. Too small and dashed aircraft are missed entirely rather than
  flagged.
- `mask.dark_sigma` (1.5) — how far below the sky a pixel must sit to count as
  foreground.
- `gradient.poly_order` (3) — see the limitation below.

## Known limitations

- **A compact light source is not removed by the gradient fit.** A low-order
  polynomial models the broad skyglow ramp, not a lamp just outside the frame.
  Raising the order enough to chase it starts eating real sky. Mask or crop the
  lamp instead.
- **Each group produces its own composite.** Frames either side of a large
  camera move cannot share one alignment, so their meteors cannot land in one
  image without reprojecting between groups, which this does not attempt.
- **The ground stack uses a sigma-clipped mean, not a median.** A median rejects
  the moving sky better but needs every frame in memory at once.
- **Detection runs on the JPEGs**, by design — speed matters more than shadow
  detail for finding streaks. The stacks use the RAF files when present.

## Rough timings

The synthetic set (40 frames at 1600×1067) runs end to end in about 70 seconds.
Full-resolution X-T30 III frames are roughly 15× the pixels, so a 174-frame
night should be on the order of an hour from the JPEGs, plus RAF decode time for
the stacks. Registration and detection dominate; both are cached.
