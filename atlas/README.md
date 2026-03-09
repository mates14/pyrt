# ATLAS ECSV Binary (Optional)

## Description

This is a modified version of the original ATLAS refcat extraction tool with added support for ECSV (Enhanced Character Separated Values) output format.

The ECSV output provides **dramatically faster loading** of catalog data into Python/Astropy compared to traditional text formats.

## Building

Simple compilation with standard tools:

```bash
make atlas
```

Or manually:
```bash
gcc -o atlas atlas.c -lm
```

## Requirements

- C compiler (gcc, clang, etc.)
- libm (math library - standard on all Unix systems)

## Installation

After building, copy the `atlas` binary to somewhere in your PATH:

```bash
# Option 1: User bin directory
cp atlas ~/bin/

# Option 2: Local bin directory
sudo cp atlas /usr/local/bin/

# Option 3: Keep in pyrt directory
# (make sure it's in PATH or pyrt will find it locally)
```

## Usage

The tool is called automatically by PYRT's catalog module when using local ATLAS catalogs:

```python
from pyrt.catalog import Catalog

# Will use local atlas binary if available
cat = Catalog("atlas@localhost", ra=123.456, dec=45.678, radius=0.1)
```

## Note

This tool is **completely optional**. PYRT can also query ATLAS via VizieR:

```python
cat = Catalog("atlas@vizier", ra=123.456, dec=45.678, radius=0.1)
```

The local version is faster for repeated queries if you have local ATLAS refcat files.

## Modifications

This version differs from the original ATLAS refcat tool by adding ECSV output format support for efficient Python integration.

### Bug fix: tile-selection "chord through interior" (2026-03)

The tile-selection algorithm used a two-step approach to determine which 1°×1° sky tiles to read:

1. Mark tiles containing the rectangle corners.
2. Mark tiles whose own corners fall inside the rectangle.

This failed for rectangles that are narrow in Dec (or RA) and fit entirely within a single 1° coordinate band. In that case no tile corners fall inside the rectangle, so step 2 marks nothing and only the two end tiles from step 1 are read — all intermediate tiles are silently skipped.

**Symptom**: querying a field spanning, e.g., RA 92°–101° at Dec ~73.5° returned stars only from tiles `092+73` and `101+73`; tiles `093+73` through `100+73` were omitted, reducing the star count by ~90 %.

**Fix**: after step 2, a fill-between pass is applied in both directions — for each Dec band, all tiles between the westernmost and easternmost marked tile are filled in, and vice-versa for each RA column. This covers rectangles that are narrow in either dimension.

**RA wrap-around sub-case**: if the field straddles RA=0° (e.g., RA 356°–4°), the two extreme marked tiles are at i=356 and i=4. A naive min-to-max fill would cover 352° the wrong way. The fix detects this when `ramax − ramin > 180` and instead fills the short arc: `[ramax..359]` + `[0..ramin]`.

**Polar field exclusion**: when the field contains a celestial pole, `adoffset()` reflects corners that would cross the pole to RA+180°. These antipodal corner positions would seed the fill-between with bogus extremes. The fill is therefore skipped entirely for polar fields. This is safe because near the pole `cos(Dec)→0`, making the RA angular constraint trivially satisfied; step 2 already marks all tiles in the high-Dec bands without needing the fill.
