# Per-camera parameters (`cameras.ini`)

pyrt itself contains no instrument names or instrument numbers. Everything that
belongs to a particular camera lives in one INI file and is chosen per frame from
the FITS header:

- ZPN camera models
- ASTQUAL limits
- detector gain, read noise and saturation
- dark noise models

A camera the file does not know still works: it gets the generic `[DEFAULT]`
values and the generic code paths (header WCS, stepwise term selection).

This document has two parts. The **reference** explains how the system works and
how to edit it. The **record** (from "What changed" on) lists what was done in
October 2026, which behaviour changed on purpose, and what is still open.

---

## Reference

### Files

| File | Role |
|---|---|
| `pyrt/utils/chip_id.py` | Camera registry: FITS header → short camera ID (`andor46`, `mi6166`, …). The same registry asarina's calibration uses. CLI: `pyrt-chip-id`. |
| `pyrt/utils/camera_config.py` | Resolver: header → merged parameters, with the origin of each value (`CameraParams`). |
| `pyrt/utils/cameras.ini` | The parameters, shipped with pyrt and versioned in git. |
| `~/.config/pyrt/cameras.ini` | Optional local additions and overrides (same format). |

The registry moved to pyrt from `asarina/asarina/chip_id.py`. asarina now
imports `pyrt.utils.chip_id`; the old file is deleted.

### How a frame finds its parameters

**1. Three keys come from the header:**

| Key | Source |
|---|---|
| camera | `CAMERA_ID` if present (cat2det writes it into every `.ecsv`), otherwise the registry run on the header (`CCD_SER`, `CCD_TYPE`, `INSTRUME`, …). If the registry doesn't recognise the camera — `unknown`, or only a family such as `mi` — the camera becomes `ccd:<CCD_NAME>`. |
| telescope | `TELESCOP`, compared as an exact string (e.g. `D50`, `SBT`, `SBT sky monitor`, `FRAM at Pierre Auger Observatory`). |
| date | the first 10 characters of `DATE-OBS` (`YYYY-MM-DD`). |

**2. Sections that match are merged, least specific first:**

```
[DEFAULT]                      every camera
[mi6166]                       this camera, wherever it is mounted   (detector)
[mi6166/2026-10-02]            ... from that date on
[mi6166@SBT]                   this camera on this telescope         (optics)
[mi6166@SBT/2026-10-02]        ... from that date on
```

- **More specific wins.** A later layer overrides keys of an earlier one.
- **Dated sections add up.** Every dated section with date ≤ DATE-OBS applies, in
  date order. A recalibration therefore lists only what changed. An undated
  section counts as "from the beginning".
- **Missing date.** A frame without `DATE-OBS` gets no dated sections.
- **Missing telescope.** A frame without `TELESCOP` gets no `@telescope` sections,
  only `[DEFAULT]` and `[camera]`.
- **Several cameras per section.** One section may name several cameras:
  `[andor46,fli534@D50]`.
- **Cameras the registry doesn't know** use the `ccd:` form: `[ccd:SROT1]`.
- **Malformed names.** A section name that doesn't parse is ignored, with a warning.

**3. Two files.** The shipped file is read first, then `~/.config/pyrt/cameras.ini`.

- Specificity comes first, file order second. Inside the *same* section, the user
  file wins.
- A less specific section in the user file does **not** beat a more specific one
  in the shipped file. To override a value, use the same section name as the
  shipped file.

**Worked example.** Take an SBT C1 frame from 2026-10-05: `CCD_SER = G4KF16000-06166`,
`TELESCOP = SBT`. It resolves to camera `mi6166` and applies
`[DEFAULT]` → `[mi6166]` → `[mi6166@SBT]`. Its `CAMCONF` is `mi6166@SBT`.

### Values the header can also carry

Gain and saturation can appear both in the header and in `cameras.ini`.
`CameraParams.resolve()` takes them in this order:

```
command line  >  force.<key> in cameras.ini  >  header keyword  >  <key> in cameras.ini
```

- A plain `gain = …` is a **fallback** for frames that have no `GAIN`.
- `force.gain = …` **overrides** the header. Use it for a camera whose driver writes
  a wrong value.
- If the header holds a value that isn't a number, it is skipped with a warning.

### Keys

**Camera model**, used by `pyrt/core/refit_astrometry.py`:

| Key | Meaning |
|---|---|
| `proj` | Projection, default `ZPN`. ZPN always gets `PV2_1 = 1` fixed. |
| `PV2_3`, `PV2_5`, … | Radial terms. |
| `CRPIX1`, `CRPIX2` | Optical axis in **full-frame unbinned** pixels. Converted to sub-frame pixels with `LTV1/2`, `LTM1_1/2_2` when the header has them. |
| `always_fit` | `yes` = fit the model even without `-z`. Used only by NF4. |

How the code uses them:

- **A camera "has a model"** when its sections define any `PV2_n` or `CRPIX` key.
  This replaces the old lists of known camera names.
- **With `-z`**, the model is the starting point and is fitted. If the header
  already holds a ZPN solution, that solution's values are the start instead.
  This header seeding behaves as before.
- **With `--zpn-from-camera` and no `-z`**, the model is held fixed and only CRVAL
  and CD are fitted. If the camera has no model, a warning is logged and the
  header WCS is used.
- **With `-z` and no model**, stepwise term selection and projection selection run,
  as for any unknown camera.

**ASTQUAL limits**, in `[DEFAULT]` and overridable per camera:

| Key | Meaning | Default |
|---|---|---|
| `astqual_max_dscal` | \|final/input pixel scale − 1\| | 0.02 |
| `astqual_max_aniso` | CD singular value ratio − 1 | 0.02 |
| `astqual_max_drot` | rotation change against input WCS (deg) | 1.0 |
| `astqual_max_dcen` | chip centre shift against input WCS (px) | 10.0 |
| `astqual_max_dmax` | max displacement over the chip (px) | 20.0 |
| `astqual_jac_min`, `astqual_jac_max` | local scale relative to centre | 0.9, 1.1 |
| `astqual_max_sigf` | ASTSIGMA / FWHM | 0.5 |
| `astqual_min_cover` | fraction of 4×4 cells with a fitted star | 0.75 |
| `astqual_max_chi2c` | χ²/dof of per-cell mean residuals | 3.0 |
| `astqual_sys_fwhm` | tolerated systematic cell shift (FWHM fraction) | 0.04 |

**Detector**, used by `pyrt/cli/phcat.py` and `pyrt/cli/dophot.py`. All three
values are **per single readout**.

| Key | phcat | dophot `-w` weight image |
|---|---|---|
| `gain` | `-g` > `force.gain` > header `GAIN` > `gain`. Values from `cameras.ini` are multiplied by `NCOMBINE`; `-g` and header values are used as they are. | `-G` > `force.gain` > `GAIN` > `gain` (× `NCOMBINE`). |
| `rnoise` | `force.rnoise` > `rnoise`, × √ACCNUM / √NCOMBINE. Never read from the header. | — |
| `saturation` | header `SATURATE` > `saturation` × ACCNUM. Also used for SExtractor's `SATUR_LEVEL`. | — |

**Dark noise model**, used by asarina `make_calib` to validate darks by pair
differences:

```
noise.<binning>            = rn=.. [dark=.. t0=.. tdouble=..] tol=..
noise.<binning>.<readmode> = ...       (cameras whose readout modes differ)

sigma = sqrt(rn² + dark·(exptime + t0) · 2^(temp/tdouble))
```

- Without `tdouble` there is no temperature factor.
- `tol` is the relative tolerance.
- **Fallbacks.** A missing binning falls back to the camera's `noise.1x1`. A
  read-mode camera asked for an unlisted mode, or a camera with no model, gets
  `[DEFAULT]`.
- **Only `[DEFAULT]` and undated `[camera]` sections apply here.** make_calib looks
  cameras up by ID only, with no telescope and no date, so noise entries placed in
  `@telescope` or dated sections are never seen.

### Provenance: what was used

- **`CAMCONF`** in the WCS header, written by both header writers in `zpnfit`. It
  names the most specific section applied, e.g. `andor46,fli534@D50/2026-08-01`, or
  `<camera>:none` if no camera section matched.
- **phcat's log line** gives the origin of each value, e.g.
  `epadu=0.81 (header) rnoise=4.63 (cameras.ini[andor46])`.
- **In Python:** `CameraParams.from_header(h).source[key]` gives `file[section]`
  for any key.

### Common tasks

**Which camera is this file?**

```
pyrt-chip-id -v frame.fits          # camera ID and the header field that matched
pyrt-chip-id --list                 # all known cameras
pyrt-chip-id --info mi6166          # patterns for one camera
```

**What will this frame get?**

```python
from astropy.io import fits
from pyrt.utils.camera_config import CameraParams
p = CameraParams.from_header(fits.getheader('frame.fits'))
print(p)                 # camera@telescope date, sections applied
print(p.values)          # merged key -> value (strings)
print(p.source)          # key -> file[section]
```

**New camera.**

1. Add its header patterns to `CAMERA_PATTERNS` in `pyrt/utils/chip_id.py`.
   Optionally add a description to `CAMERA_DESCRIPTIONS`.
2. Add `[newcam]` (detector values) and/or `[newcam@TELESCOP]` (camera model) to
   `cameras.ini`.
3. Check the result with the snippet above.

**Recalibration.** Add a dated section that lists only the changed keys, e.g.
`[mi6166@SBT/2027-03-01]`. Older frames keep the older values. Keep the
calibration notes as comments above the section.

**Camera moved to another telescope.** Add `[cam@NEWTEL]`. Its detector section
`[cam]` goes along automatically.

**Local experiment without editing the repo.** Put the same section name in
`~/.config/pyrt/cameras.ini`.

**Caution.** Keys are not validated, so a misspelt key (`PV2-3`, `astqual_max_dcn`)
is silently ignored. Check with `p.values` after editing.

---

## Record: what was done (October 2026)

### Commits

| Repo | Commit | Content |
|---|---|---|
| pyrt | `58fe1a2` | chip_id registry moved into pyrt; `pyrt-chip-id`; cat2det writes `CAMERA_ID`; new registry ID `mi620x` |
| pyrt | `103a586` | `camera_config.py`, `cameras.ini`; refit_astrometry camera models and ASTQUAL limits from the file; `CAMCONF` |
| pyrt | `ddc512a` | phcat and dophot gain, read noise and saturation from the file; dophot `gain = 2.3` default removed |
| pyrt | `203a86f` | dark noise models in `cameras.ini` |
| asarina | `870a694` | imports `pyrt.utils.chip_id` |
| asarina | `b182a37` | make_calib reads the noise models from `cameras.ini`; `asarina/chip_id.py` deleted |

### What moved where

| Before (hardcoded) | Now |
|---|---|
| `refit_astrometry.py`: ZPN priors chosen by `CCD_NAME` (C0, C1, C2, makak, makak2, NF4, SROT1, ASM1, ASM-S, CAM-ZEA) and the camera list in five copies | sections `[andor46,fli534@D50]` (+ `/2026-08-01`), `[mi6166@SBT]`, `[mi6167@SBT]`, `[mi0315@SBT sky monitor]`, `[mi620x@FRAM at Pierre Auger Observatory]`, `[ccd:SROT1]`, `[ccd:ASM1]`, `[ccd:ASM-S]`, `[ccd:CAM-ZEA]` |
| `refit_astrometry.py`: `QUAL_*` constants | `astqual_*` in `[DEFAULT]` |
| `phcat.py`: 0.81 / 4.63 / 60000 (the D50 Andor values, used for every camera) | `[andor46]`, and also `[DEFAULT]` so nothing changes for other cameras |
| `dophot_config.py`: `gain = 2.3` | removed (header / `cameras.ini`) |
| asarina `make_calib.py`: `CAMERA_NOISE_MODELS` lambdas, `DEFAULT_NOISE_MODEL` | `noise.*` keys in camera sections and `[DEFAULT]` |

Mapping from the old `CCD_NAME` keys:

- **D50 C0** → `andor46` and `fli534` at `D50`. The FLI shares the Andor's
  values until it is fitted on its own data.
- **SBT C1, C2** → `mi6166`, `mi6167` at `SBT`.
- **makak** → `mi0315` at `SBT sky monitor`.
- **makak2** is a cropped makak with identical headers, so it resolves to `mi0315`
  as well; the crop is handled by LTV/LTM.
- **NF4** → `mi620x`, the G4-16000 whose serial is truncated to
  `G4KF16000-0620`, at FRAM.
- **SROT1** (one ROST test image), **ASM1, ASM-S** (BOOTES all-sky tests) and
  **CAM-ZEA** are not in the registry, so they keep `ccd:` sections.

### Behaviour that changed on purpose

1. **D50, one model per date, whatever the mode.** Before, `-z` started from
   PV2_3 = 260 with CRPIX (548, 565), while `--zpn-from-camera` without `-z` held
   the older 300 with (543, 530). Now frames before 2026-08-01 get 300 / (543, 530)
   in both modes, and frames from 2026-08-01 on get 260 / (548, 565). On a frame
   from 2026-10-12, `--zpn-from-camera` improved ASTSIGMA from 0.082 to 0.062 px;
   `-z` gave identical results to the old code.
2. **The SBT models now apply only to the SBT cameras.** Before, `CCD_NAME = C1` got
   the SBT C1 model on any telescope, including the D50 wide-field FLI (`fli785`)
   and the OAN 1.5 m camera. Those are now unknown cameras: with `-z` they get
   stepwise selection.
3. **dophot `-w`** (weight image) used gain 2.3 regardless of the header. It now
   uses `-G`, the header `GAIN`, or `cameras.ini`, in that order.

Everything else was verified to be unchanged:

- refit model setup: 16 header cases × 4 option combinations, before vs after
- phcat gain, read noise and saturation: four real frames (D50 single scan, D50
  kinetic series, SBT C1, makak stack)
- make_calib noise: 1485 camera × binning × exposure × temperature × read-mode
  combinations
- NF4 is still fitted without `-z`

### Deployment

asarina now reads parameters from pyrt's `cameras.ini`, so **update pyrt before
asarina** on every machine. With an older pyrt, make_calib doesn't fail: every
camera quietly gets the 8.0 ADU default noise model, and dark validation is wrong.

### Open items

- **D50 2×2 frames have no `LTM`/`LTV` keywords**, so the full-frame CRPIX priors
  (548, 565) are applied unscaled to the 512×512 frame. This predates the change.
  It matters for `--zpn-from-camera`; `-z` may recover from it. A fix would be to
  fall back to `XBINNING`/`YBINNING` when LTM is missing.
- **Read noise units.** Is 4.63 in electrons or ADU? phcat divides it by
  √NCOMBINE, which is only right for ADU, but IRAF `phot` expects electrons.
  - Kinetic-series frames (100-frame averages, `GAIN = 81`, `KINNUM = 100`) don't
    scale it at all.
  - `get_accnum()` looks for `ACQMODE = 'kinetics'`, but headers say
    `'Kinetic Series'`.
- **asarina `make_calib.py` `CAMERA_FILTERS`** (per-camera dark/flat limits) and its
  two inline fallback filters are still hardcoded. They could become
  `dark.max_sigma`, `flat.min_median`, … keys.
- **`combine.py`** keeps its own fallbacks: `-g 2.3`, and read noise √(3·gain).
- **`astqual_max_dcen` / `astqual_max_dmax` are in pixels**, calibrated on a 1k
  chip. Revisit them when the ASTQUAL thresholds are calibrated.
- **`--szp`** still adds a camera's PV2 terms to the AZP model. This predates the
  change and was left as it was.
- **fli534 has no model of its own** (it shares the Andor's); NF4's geometry is
  not settled yet.
- **Stale `CAMERA_ID`.** If the registry changes, `.ecsv` files written earlier
  keep their old `CAMERA_ID`, which wins over the registry. Rerun cat2det, or
  delete the keyword, to re-identify them.
- **Stray registry copies outside the repos:** `pyrt/pipeline/chip_id.py` and
  `pyrt/pipeline/d50/chip_id.py` (both untracked), and `~/bin/chip_id.py`.
- **No unit tests yet:** pyrt has no `tests/` directory.
