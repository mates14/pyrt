# Filter Estimation and Validation in dophot3

## The Problem

Photometric calibration depends critically on knowing the correct filter used for each observation. However, in practice, filter information can be:

1. **Missing** - No filter information in FITS header
2. **Wrong** - RTS2 and other systems occasionally write incorrect filter metadata
3. **Ambiguous** - Similar filters (Johnson_R vs Sloan_r) may be hard to distinguish

Incorrect filter assignment leads to **systematic calibration errors** that contaminate lightcurves with instrumental effects rather than true astrophysical signals.

## Current Architecture

### Two-Phase Filter Determination System

1. **Phase 1: Heuristic Matching (`determine_filter`)**
   - Exact filter name matching against catalog
   - Wavelength-based fallback for unknown filters
   - Purpose: Initial assignment for catalog loading
   - Always runs early in pipeline

2. **Phase 2: Statistical Validation/Discovery (`compute_zeropoints_all_filters`)**
   - Correlation-based filter validation/discovery
   - Tests all available catalog filters using zeropoint fitting
   - Purpose: Data-driven validation or discovery based on `--filter-check` mode
   - Only runs during photometric fitting when enabled

### Processing Flow

```
FITS Header → determine_filter() → Catalog Loading → compute_zeropoints_all_filters() → Final Calibration
     ↓              ↓                     ↓                           ↓                        ↓
  FILTER      PHFILTER,PHSYSTEM      Loaded catalog         Final filter choice         det object updates
             PHSCHEMA                                      (based on mode)
```

## Fundamental Limitations

### Cases Where Detection is Difficult/Impossible

1. **Short Exposures**: Poor photometric statistics make filter discrimination unreliable
2. **Similar Filters**: Johnson_R vs Sloan_r may be indistinguishable within noise
3. **Unfiltered Images**: Detecting clear/no-filter situations
4. **Sparse Color Coverage**: Single filter observations provide limited discriminatory power

### The Spectral Break Problem

Certain filter transformations (e.g., Gaia BP-RP-G → J-band) are fundamentally problematic due to:
- **Paschen break** around 820nm limiting infrared extrapolation
- **Spectral coverage gaps** between optical and near-IR catalogs

## Implemented Solution: Unified Filter Validation

### Command Line Interface
```bash
--filter-check={none|warn|strict|discover}
```

Short forms also accepted: `n`, `w`, `s`, `d`

### Filter Check Modes

#### `none` (default)
- **Behavior**: Trust FITS header completely, no validation
- **Use case**: Well-behaved data with reliable filter metadata
- **Performance**: Fastest, no additional computation

#### `warn` 
- **Behavior**: Validate header filter against statistical best fit, warn on mismatch but continue
- **Use case**: Operational pipelines where you want to detect problems but not halt processing
- **Output**: Warning messages for manual review

#### `strict`
- **Behavior**: Validate header filter, abort processing on mismatch
- **Use case**: Critical calibration work where filter accuracy is essential
- **Output**: Error message and program termination on mismatch

#### `discover`
- **Behavior**: Ignore FITS header completely, use statistical analysis to determine best filter
- **Use case**: Unknown filter images, completely unreliable headers, or filter identification research
- **Output**: Reports discovered filter vs. header value

### Examples

```bash
# Trust header (fastest, default)
pyrt-dophot image.fits

# Validate and warn about problems  
pyrt-dophot --filter-check=warn image.fits

# Strict validation for critical work
pyrt-dophot --filter-check=strict image.fits

# Discover filter for unknown images
pyrt-dophot --filter-check=discover mystery_image.fits

# Short form
pyrt-dophot --filter-check=d image.fits
```

### Option B: Consistency Tracking (Rejected)
Track filter consistency across observation sequences and flag sudden changes.

**Rejected because**: Too complex for an already complicated system, difficult to implement reliably.

### Option C: Model-Based Detection
Use SC (sinusoidal color) terms as a "filter position detector":

1. When no color terms are active in the model
2. Silently add SC term to fitting
3. Check if SC coefficient is unexpectedly large
4. Large SC suggests filter mismatch (image is "between" expected filters)

**Pros**: Automatic, integrated into existing fitting framework
**Cons**: Requires detecting when color terms are inactive, somewhat indirect

## Implementation Status

### ✅ Completed: Unified Filter Validation System

#### Core Features
- **Implementation**: Single `--filter-check` parameter with four modes
- **Integration**: Built into main photometric fitting pipeline via `compute_zeropoints_all_filters()`
- **Validation**: Uses correlation-based statistical analysis across all available filters
- **User Control**: Clear escalation path from none → warn → strict → discover
- **Metadata Propagation**: Updates det objects to ensure output consistency when filter changes
- **Mixed Filter Detection**: Consistency checking across multi-image datasets

#### Key Improvements Over Previous System
1. **Unified Architecture**: Single function handles both zeropoint computation and filter discovery
2. **Efficient Discovery**: Correlation-based analysis much faster than full photometric fitting
3. **Consistent Metadata**: All metadata structures (det objects, metadata list) stay synchronized
4. **Scientific Accuracy**: Proper AB→Vega conversions with Cohen et al. (2003) standards
5. **Robust Schema Design**: Eliminated mixed AB/Vega color systems that caused calibration errors

#### Bug Fixes Included
- Fixed photometric system determination to use catalog metadata instead of name-based guessing
- Fixed GaiaJ schema to avoid mixed AB/Vega systems 
- Added proper J_Vega filter with correct AB→Vega conversion (-0.894 mag)
- Ensured det object updates propagate filter changes to output files

### 🔄 Future Enhancements: Model-Based Detection (Option C)
1. Detect when color terms are not in active model
2. Add SC term during fitting as diagnostic  
3. Flag cases where SC coefficient exceeds threshold
4. Could be integrated into stepwise regression logic

### 🔄 Future Improvements: User Experience
1. Export filter confidence metrics to output files
2. Tools for batch analysis of filter consistency across sequences
3. Integration with quality control databases

## Design Philosophy

We acknowledge that **perfect automatic filter detection is impossible** in the general case. Instead, we provide:

1. **Robust defaults** that work for well-behaved data
2. **Diagnostic tools** to identify problematic cases  
3. **User control** over validation strictness
4. **Clear documentation** of limitations and assumptions

The goal is to **flag suspicious cases for human review** rather than attempt fully automatic correction, which could introduce subtle systematic errors.

## Technical Notes

### AB vs Vega System Handling
- Each catalog filter specifies its photometric system ('AB' or 'Vega')
- Mixed-system color terms (e.g., Johnson_I - Sloan_r) are avoided in schema design
- Standard AB→Vega transformations: J+0.9, H+1.4, K+1.9 magnitudes

### Schema Design Principles
- Pure photometric systems within each schema (no AB/Vega mixing)
- Use duplicate filters (e.g., I,I) to create natural zero colors when needed
- Wavelength-based matching for cross-catalog compatibility

### Filter Validation Metrics
- **Correlation coefficients** between observed and catalog magnitudes
- **Zeropoint scatter** (median absolute deviation from median)
- **Total star count** for statistical reliability
- **Cross-image consistency** for multi-image datasets

## References

- Standard broadband filter transformations
- RTS2 observatory control system documentation
- Photometric system definitions (AB, Vega, Johnson, SDSS)