# pyrt
Python photometric reduction tools - a suite centered around the dophot3 calibration engine

## Overview

This is a **full image calibration tool** that provides universal, automatic, and reliable calibration for astronomical observations. It performs:

- **Photometric calibration**: Converts instrumental magnitudes to calibrated photometry using all-sky reference catalogs
- **Astrometric refinement**: Significantly improves WCS solutions beyond initial astrometry.net results

Currently in production use at:
- **D50 and SBT telescopes**: Automated processing pipeline
- **Makak**: Wide-field sky monitoring camera ([lascaux.asu.cas.cz/m/makak](https://lascaux.asu.cas.cz/m/makak)) providing real-time photometric quality measurements at Ondřejov Observatory
- **General-purpose**: Tool for calibrating diverse astronomical images, handling even challenging data

**Getting Started**: Just give it an astrometry.net-solved FITS file:
```bash
dophot3.py your_image.fits
```
The tool orchestrates source extraction (SExtractor + optionally IRAF) and performs photometric calibration and astrometric refinement, producing a self-contained `.ecsv` file ready for scientific analysis.

## What dophot Does (and Doesn't Do)

The pipeline integrates several specialized tools, each doing what it does best:

### Initial Processing (External Tools)
- **Initial astrometric identification**: astrometry.net provides the starting WCS solution
- **Source extraction**: SExtractor detects objects and provides centroids
- **Photometry**: IRAF performs precision photometry (SExtractor-only mode also available, though less trusted for flux measurements; the SExtractor+IRAF combination provides better accuracy but may be too heavy for some applications)

### What dophot Actually Does

1. **Photometric Calibration (ADU → mag/Jy)**
   - Relative in-frame calibration using photometric all-sky catalogs
   - Star-to-catalog matching with NDTree algorithm for efficient pair searching
   - **Extensive catalog support**: ATLAS (local or Vizier), PanSTARRS, GAIA, SDSS, USNO-B, Makak
   - GAIA and ATLAS provide fitted Johnson magnitude conversions alongside natural SDSS/SDSS-like passbands
   - Advanced filter response and atmospheric corrections
   - Response inhomogeneity correction ("analytical superflat")
   - Frame-by-frame spatial profile fitting for unstable conditions

2. **High-Precision Astrometric Refinement**
   - Produces more stable solutions than astrometry.net's SIP tweaking or SCAMP
   - Works reliably even with 180° all-sky frames
   - Uses photometry-selected stars for superior quality
   - Production-stable: tens of thousands of images processed at SBT and D50 telescopes
   - Supports TAN/ZEA/SIN projections and SIP (up to 2nd order) out of the box
   - Note: ZPN projection requires hard-wired initial guesses (not user-configurable yet); SIP 3rd order may have numerical issues (2nd order sufficient for most cases with proper azimuthal projections)

## Architecture

```
                                FITS Image
                                    ↓
                              astrometry.net 
                                    ↓
                             WCS-solved  FITS
                                    ↓
              phcat.py: SExtractor + IRAF [or SExtractor-only]
                                    ↓
                    cat2det.py: prepare detection table
                                    ↓
        dophot3: photometric calibration + astrometric refinement
                                    ↓
                       Calibrated catalog (.ecsv)
                                    ↓
                 [photo-db archiving - not in repo yet]
```

### Components

- **Prerequisites**: astrometry.net for initial WCS solution
- **dophot3 pipeline**: Python3 + astropy + scipy (portable)
  - `phcat.py`: Orchestrates SExtractor + IRAF (or SExtractor-only mode)
  - `cat2det.py`: Prepares detection tables with FITS metadata
  - `catalog.py`: Unified catalog interface supporting multiple sources
  - `dophot3.py`: Core engine for photometric calibration and astrometric refinement
  - Generalized response model inspired by TPoint
  - **Catalog support**: ATLAS (local/Vizier), PanSTARRS, GAIA, SDSS, USNO-B, Makak
  - Johnson magnitude conversions available (GAIA, ATLAS)
  - Output: Self-contained calibrated catalog (.ecsv)
- **photo-db** (not included in this repository): PostgreSQL + q3c archiving system
  - Production deployment at D50 processing all observations
  - Cone-search queries for object retrieval
  - Time-series analysis for variable objects
  - Feeds transient search engines
  - **Status**: Currently separate from this codebase; needs to be made available

## Current Status

- Absolute photometric calibration for all observations
- Seven catalog sources supported (ATLAS, PanSTARRS, GAIA, SDSS, USNO-B, Makak)
- Johnson magnitude conversions available for GAIA and ATLAS
- Precise and reliable results
- Semi-online output (behind ASU firewall)
- Web interface available

## Usage Guide

The system is designed to be approachable while providing powerful capabilities:

### Basic Workflow
The `dophot3` tool is telescope-agnostic and works with astrometry.net-solved FITS images. Helper scripts (`phcat.py` wraps SExtractor+IRAF, `cat2det.py` prepares detection tables) are orchestrated automatically when needed. The tool performs photometric calibration and astrometric refinement, producing self-contained `.ecsv` catalogs.

You can use single-image mode for individual object photometry, or multi-image mode for advanced calibration including color responses and atmospheric modeling. Multi-pass processing is supported - output `.ecsv` files can be fed back as input for iterative refinement. The "train once, apply many" workflow lets you create robust photometric models from representative images and apply them to individual observations.

See [dophot-usage.md](dophot-usage.md) for complete workflow documentation.

### Filter Handling
Photometric calibration critically depends on correct filter identification. The system provides four validation modes: `none` (trust headers), `warn` (validate and notify), `strict` (abort on mismatch), and `discover` (determine filter statistically). This unified approach handles missing, incorrect, or ambiguous filter information while maintaining processing efficiency. The correlation-based statistical analysis can identify filters even when header metadata is unreliable. See [filter-estimation.md](filter-estimation.md) for filter validation details.

### Calibration Terms
The photometric model uses an elegant notation inspired by TPoint's telescope pointing correction syntax, built on three conceptual levels:

1. **Pure terms** - the actual mathematical terms that get fitted: `PXY`, `SC`, `RS`, `PAC`, etc.
2. **Macros** - compact notation that expands to term sets: `.p3` (all spatial terms to 3rd order), `.r2` (radial terms to 2nd order), `.l` (linearity terms)
3. **Modifiers** - prefixes that control fitting behavior: `@` (stepwise selection), `&` (always include), `#` (fixed value), `*` (per-image)

This three-level system provides a domain-specific language matching how astronomers naturally think about calibration: "let the data decide" (stepwise), "physics requires this" (always), or "we measured this separately" (fixed). See [terms-explained.md](terms-explained.md) for the complete term syntax guide.

## Future Development

### 1. Code Release
- **photo-db**: Make the PostgreSQL+q3c database system available (currently production at D50 but not in repository)
- Archive ingestion tools and query interfaces
- Documentation for database schema and deployment

### 2. Astrometry Configuration
- Make ZPN projection initial guesses user-configurable (currently hard-wired in code)
- Generalize camera parameters for easier deployment to new instruments
- Address SIP 3rd order numerical implementation (2nd order works reliably)
- Note: Standard projections (TAN/ZEA/SIN) and SIP up to 2nd order already work without configuration
- Current production deployment: 3.5°×3.5° (SBT) and 20°×30° (Makak) all-sky cameras

### 3. Catalog Performance
- Smart catalog caching to speed up repeated queries
- Optimization for large-scale survey operations

### 4. Alert System
- Automated brightening detection alerts
- Transient discovery capabilities

## Technical Notes

- **Multi-Catalog Support**: Seven catalogs currently implemented (ATLAS local/Vizier, PanSTARRS, GAIA, SDSS, USNO-B, Makak)
- **Photometric System Flexibility**: GAIA and ATLAS provide Johnson magnitude conversions; PanSTARRS uses SDSS-like passbands
- **All-Sky Catalog Calibration**: Works without strictly photometric nights by using all-sky reference catalogs
- **Automated Quality Control**: Automatic frame assessment and bad frame rejection
- **Production Deployments**: D50, FRAM, SBT, Makak all-sky monitor, Mini/Mega Tortora
- **Atmospheric Monitoring**: Makak system provides additional atmospheric calibration data

## Reference

**Jelínek, M.** (2023). "Photometric pipeline for robotic telescopes", *Contributions of the Astronomical Observatory Skalnate Pleso*, **53**(4), 127-135. [doi:10.31577/caosp.2023.53.4.127](https://doi.org/10.31577/caosp.2023.53.4.127)
