# Understanding Photometric Fitting Terms in dophot3

## The Problem We're Solving

When we photograph stars, the brightness we measure isn't exactly the true brightness because:

1. **Atmospheric extinction** - air absorbs light
2. **Instrument distortions** - cameras aren't perfect  
3. **Color sensitivity** - detectors respond differently to different star colors

## The Solution: Mathematical Model

We model these effects with a polynomial equation:
```
observed_magnitude = true_magnitude + corrections
```

Where corrections include:
- **Spatial terms** (`PX`, `PY`, `PXY`, etc.) - fix optical distortions across the image
- **Radial terms** (`PR`, `P2R`) - fix lens distortions that depend on distance from center
- **Color terms** (`PC`) - fix differences in how the instrument sees different star colors

## The Extended -U Syntax

Instead of specifying long lists of terms and options, dophot3 uses an elegant shorthand notation:

```bash
dophot3 image.fits -U '@.p2,&.r2,#PC=0.1'
```

### Modifiers: How to Handle Each Term

| Modifier | Meaning | Behavior | Use Case |
|----------|---------|----------|----------|
| `@` | Stepwise | Let statistics decide if needed | Unknown significance |
| `&` | Always | Always include, never remove | Physics requires it |
| `#` | Fixed | Lock at specific value | Known from calibration |
| (none) | Default | Use `--use-stepwise` setting | Normal operation |

### Examples Explained

#### **`@.p2`** = "Search among 2nd degree spatial corrections"
- `@` = "use stepwise regression" (let the computer decide what's needed)
- `.p2` = "2nd degree polynomial" → expands to `PX`, `PY`, `PXY`, `P2X`, `P2Y`
- **Translation**: "Try different combinations of spatial corrections and keep only the significant ones"

#### **`&.r2`** = "Always include radial corrections"
- `&` = "always use these" (don't let the computer remove them)
- `.r2` = "radial terms up to 2nd degree" → `PR`, `P2R`
- **Translation**: "We know lenses have radial distortion, so always correct for it"

#### **`#PC=0.1`** = "Fix color correction at 0.1"
- `#` = "fix this value" (don't change it during fitting)
- `PC=0.1` = "color correction coefficient = 0.1 magnitudes"
- **Translation**: "We know this instrument has a 0.1 mag color bias, keep it fixed"

## Term Types and Macros

### Spatial Polynomials (`.p`)
- **`.p1`** → `PX`, `PY` (linear trends across image)
- **`.p2`** → `PX`, `PY`, `PXY`, `P2X`, `P2Y` (quadratic distortions)
- **`.p3`** → adds `P3X`, `P2XY`, `PXY2`, `P3Y` (cubic distortions)

### Radial Polynomials (`.r`)
- **`.r1`** → `PR` (linear radial distortion)
- **`.r2`** → `PR`, `P2R` (quadratic radial distortion)
- **`.r3`** → `PR`, `P2R`, `P3R` (cubic radial distortion)

### Other 1D Polynomials
- **`.c`**, **`.d`**, **`.e`**, **`.f`**, **`.n`** - various specialized corrections
- Follow same pattern: `.c2` → `PC`, `P2C`

### Special Terms
- **`.l`** → `RC`, `RS`, `RO` (nonlinearity corrections)
- **`.s`** → `SXY`, `SX`, `SY` (subpixel variations)

## Human-Friendly Notation

The system uses mathematical convention where:
- **Coefficients of 0 are omitted** (no term appears)
- **Exponents of 1 are omitted** (`X¹` → `X`, `Y¹` → `Y`)
- **Only significant powers are shown**

Examples:
- `P0X1Y` → `PY` (0 coefficient for X, exponent 1 for Y)
- `P1X0Y` → `PX` (exponent 1 for X, 0 coefficient for Y)
- `P1X1Y` → `PXY` (both exponents are 1)
- `P2X3Y` → `P2X3Y` (both exponents > 1, keep them)

## Complete Examples

### Example 1: Basic Spatial Correction
```bash
-U '@.p2'
```
**Result**: Tests `PX`, `PY`, `PXY`, `P2X`, `P2Y` and keeps statistically significant terms.

### Example 2: Conservative Approach
```bash
-U '&.r2,&PC'
```
**Result**: Always includes `PR`, `P2R`, `PC` regardless of statistics.

### Example 3: Mixed Strategy
```bash
-U '@.p3,&.r2,#PC=0.1,SC=0.05'
```
**Result**: 
- Stepwise selection from 3rd degree spatial terms
- Always include radial terms `PR`, `P2R`
- Fix primary color term at 0.1
- Set initial value for secondary color term, then use default behavior

### Example 4: Complex Calibration
```bash
-U '#PC=0.12,#PD=0.03,&.r3,@.p4,SX=0.001'
```
**Result**:
- Fix known color corrections from previous calibration
- Always include 3rd degree radial terms (lens distortion)
- Search among 4th degree spatial terms
- Start subpixel term with initial value 0.001

## Default Behavior Control

The `--use-stepwise` option controls unmarked terms:

```bash
# Stepwise by default (default: --use-stepwise=True)
-U '.p2,.r2'          # Both use stepwise selection

# Direct by default (--use-stepwise=False)  
-U '.p2,.r2'          # Both always included

# Mixed with explicit control
-U '@.p2,&.r2'        # .p2 uses stepwise, .r2 always included
```

## Why This Design Makes Sense

The notation maps directly to how astronomers think about calibration:

1. **`@` (stepwise)** = "Let the data decide" - statistical approach
2. **`&` (always)** = "Physics requires this" - theoretical knowledge
3. **`#` (fixed)** = "We measured this separately" - empirical knowledge

Instead of writing:
```bash
--fit-terms="PX,PY,PXY,P2X,P2Y,PR,P2R" --fix-terms="PC" --initial-values="PC=0.1" --stepwise-for="PX,PY,PXY,P2X,P2Y" --always-fit="PR,P2R"
```

We write:
```bash
-U '@.p2,&.r2,#PC=0.1'
```

This is not cryptic code - it's **domain-specific language** that perfectly captures the three fundamental approaches to astronomical calibration: statistical selection, physical requirements, and empirical measurements.

## Priority Rules

When terms overlap, the system follows logical precedence:

1. **Fixed terms** (`#`) override everything
2. **Direct terms** (`&`) override stepwise and default  
3. **Stepwise terms** (`@`) override default
4. **Default behavior** applies to unmarked terms

Example: `-U '#PC=0.1,&.r2'` where `.r2` includes `PC`
- Result: `PC` is fixed at 0.1, `PR` and `P2R` are always selected
- No duplication or conflict

This ensures that your scientific intent is preserved regardless of how you specify overlapping term sets.