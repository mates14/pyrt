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

Instead of specifying long lists of terms and options, pyrt-dophot uses an elegant shorthand notation:

```bash
pyrt-dophot image.fits -U '@.p2,&.r2,#PC=0.1'
```

### Modifiers: How to Handle Each Term

| Modifier | Meaning | Behavior | Use Case |
|----------|---------|----------|----------|
| `@` | Stepwise | Let statistics decide if needed | Unknown significance |
| `&` | Always | Always include, never remove | Physics requires it |
| `#` | Fixed | Lock at specific value | Known from calibration |
| `*` | Per-image | Fit separately for each image | Variable conditions |
| (none) | Default | Use `--use-stepwise` setting | Normal operation |

**Note**: Modifiers can be combined! For example, `&*PX` means "always fit PX separately for each image".

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

#### **`&*PX`** = "Always fit X-direction sky transparency separately for each image"
- `&` = "always include" (don't let stepwise remove it)
- `*` = "per-image" (fit different value for each image)
- `PX` = "X-direction transparency gradient"
- **Translation**: "Passing clouds create transparency gradients that vary between exposures"

## Term Types and Macros

### Spatial Polynomials (`.p`)
- **`.p1`** → `PX`, `PY` (linear gradients: flatfield imperfections + transparency trends)
- **`.p2`** → `PX`, `PY`, `PXY`, `P2X`, `P2Y` (quadratic terms: vignetting + complex cloud patterns)
- **`.p3`** → adds `P3X`, `P2XY`, `PXY2`, `P3Y` (higher-order: physicist's magic for complex effects)

**What spatial terms actually correct:**
1. **Flatfield gradients** (stable): Twilight sky is never perfectly uniform
2. **Sky transparency variations** (per-image): Passing clouds, especially complex patterns on large fields
3. **Optical effects**: Vignetting, field curvature, and other instrumental signatures
4. **Unknown systematics**: Higher-order polynomials catch what we don't understand

### Radial Polynomials (`.r`)
- **`.r1`** → `PR` (linear radial effects)
- **`.r2`** → `PR`, `P2R` (quadratic radial effects)
- **`.r3`** → `PR`, `P2R`, `P3R` (cubic radial effects)

**What radial terms actually correct:**
1. **Aperture effects**: Optical aberrations cause different light collection efficiency vs. radial distance
2. **Flatfield over/under-correction**: Twilight flats contain diffuse scattered light that doesn't affect point sources
3. **PSF variations**: Seeing changes with field position due to atmospheric/optical effects

### Color and Atmospheric Polynomials

pyrt-dophot loads **five magnitude filters** that create **four colors** for photometric modeling:

#### Color Variables (C, D, E, F)
- **C** = color1 = (2nd filter - 1st filter) = canonically (2nd bluest - bluest)
- **D** = color2 = (3rd filter - 2nd filter)
- **E** = color3 = (4th filter - 3rd filter)
- **F** = color4 = (5th filter - 4th filter)

Any polynomial combination is possible: `PC`, `P2C`, `PD`, `PC2D`, `PCDF`, etc.

#### Universal Color Terms
- **`PC`** = Standard color term using color C (most common)
- **`XC`** = Simple universal color correction
- **`SC`** = Sophisticated universal color term for unknown responses

#### Atmospheric Variables
- **A** = Airmass (absolute - total atmospheric path length)
- **B** = Airmass (relative to image center - differential extinction)
- **N** = Instrumental magnitude (for nonlinearity)

Examples:
- **`PA`** = Linear absolute airmass correction (for all-sky work)
- **`PAC`** = **Atmospheric reddening** - how atmosphere differentially absorbs colors (red vs blue)
- **`PB`** = Linear relative airmass correction (⚠️ correlates with PX/PY - needs large fields + good airmass coverage)
- **`PBC`** = Relative airmass color term

#### ⚠️ Nonlinearity Warning
- **`PN`, `P2N`** = Polynomial nonlinearity terms - **DISCOURAGED!** Tends to fail spectacularly
- **`.l`** = Better approach → `RC`, `RO`, `RS` terms designed specifically for nonlinearity

**Note**: Both absolute (A) and relative (B) airmass are available. PB terms practically always correlate with PX/PY spatial terms, so they're only useful for large field images with multi-image photometry covering good airmass range. PA terms are designed for all-sky work and single-zeropoint modeling (-Z option).

### Special Terms
- **`.l`** → `RC`, `RS`, `RO` (nonlinearity corrections)
- **`.s`** → `SXY`, `SX`, `SY` (subpixel variations)

## Per-Image Terms: When Each Photo is Different

Sometimes, conditions change between photos. Maybe the telescope was refocused, the weather changed, or the mount shifted slightly. For these cases, you want to fit different correction values for each image.

### The Magic `*` Modifier

Adding `*` to any term makes it **per-image**:
- `PX` = one X-tilt for all images (global)
- `*PX` = different X-tilt for each image (per-image)

**How it works internally:**
- `*PX` with 3 images becomes `PX:1`, `PX:2`, `PX:3`
- Each gets its own fitted value: maybe +0.02, -0.01, +0.05
- The system automatically displays this in a nice table

### The `-y` Option: Legacy Made Easy

The `-y` option is a time-honored tradition that means "fit transparency gradients per-image":

```bash
pyrt-dophot -y images*.fits     # Old reliable way
```

**What happens under the hood:**
- Automatically adds `&*.p` to your terms
- Expands to `&*PX,&*PY` = "always fit X and Y transparency gradients per-image"
- Compatible with any other terms you specify

**Why this makes sense:**
- Sky transparency varies due to passing clouds
- Each exposure sees different atmospheric conditions
- Flatfield gradients are stable, but cloud patterns change
- **Solution**: Let each image have its own transparency correction

### Combining Per-Image with Other Modifiers

```bash
# Always fit focus per-image, but let stepwise decide about distortions
-U '&*P2R,@.p2'

# Fix the global color term, fit per-image X/Y corrections
-U '#PC=0.12,&*PX,&*PY'

# Use -y for pointing, add custom per-image focus term
pyrt-dophot -y -U '*P2R' images*.fits
```

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

### Example 5: Variable Conditions (Multi-Night Data)
```bash
pyrt-dophot -y -U '@.p2,&*P2R,#PC=0.1' night1/*.fits night2/*.fits
```
**Result**:
- `-y` adds per-image transparency corrections (each image gets its own PX, PY for clouds)
- `@.p2` searches among 2nd degree distortions (applied globally)
- `&*P2R` fits radial correction per-image (frost/dew formation or changing aperture effects)
- `#PC=0.1` keeps color correction fixed (instrument property)

**Translation**: "Sky transparency varies between exposures due to clouds, radial effects evolve through the night (frost/dew on optics), but the overall distortion pattern and color response stay stable."

### Example 6: Compact Per-Image Syntax
```bash
-U '&*.p2'              # Same as: &*PX,&*PY,&*PXY,&*P2X,&*P2Y
```
**Result**: All 2nd degree spatial terms fitted separately for each image.
**Use case**: Large field images with complex, varying cloud patterns across exposures.

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

## Priority Rules: When Terms Conflict

The system automatically resolves conflicts using logical precedence - no cryptic error messages, just sensible behavior:

### 1. Fixed Terms Rule: `#` Wins Everything
```bash
-U '#PC=0.1,&PC,@PC'      # Result: PC fixed at 0.1 (others ignored)
```

### 2. Per-Image Beats Global Rule: `*` Wins Over Non-`*`
```bash
pyrt-dophot -y -U '.p2'       # Result: Per-image PX,PY from -y; global PX,PY removed
-U '&*PX,@PX'            # Result: Per-image PX always fitted; global PX ignored
```
**Why**: If you're fitting something per-image, you obviously don't want a global version too.

### 3. Explicit Beats Default Rule: `&` and `@` Win Over Unmarked
```bash
-U '&PC,@PX,.p1'         # Result: PC always, PX stepwise, PY uses default
```

### 4. Direct Beats Stepwise Rule: `&` Wins Over `@`
```bash
-U '&PC,@PC'             # Result: PC always fitted (stepwise ignored)
```

### Real-World Example: Conflict Resolution in Action
```bash
pyrt-dophot -y -U '@.p3,&.p2,#PXY=0.02' images*.fits
```

**What the system does automatically:**
1. `-y` wants per-image `PX:1, PY:1, PX:2, PY:2, ...`
2. `@.p3` wants stepwise selection from `PX, PY, PXY, P2X, P2Y, P3X, P3Y, PX2Y, P2XY`
3. `&.p2` wants direct fitting of `PX, PY, PXY, P2X, P2Y`
4. `#PXY=0.02` wants PXY fixed

**Smart resolution:**
- ✅ `PX, PY` → per-image wins (from `-y`), global versions removed
- ✅ `PXY` → fixed wins (value = 0.02), other requests ignored
- ✅ `P2X, P2Y` → direct wins (from `&.p2`)
- ✅ `P3X, P3Y, PX2Y, P2XY` → stepwise candidates (from `@.p3`)

**Translation**: "Fit pointing per-image, fix the cross-term, force 2nd order, and let statistics decide about 3rd order."

No conflicts, no duplicates, no confusion - just exactly what you meant!

## Quick Reference: The Essential Patterns

### For Beginners (Start Here!)
```bash
# Basic spatial correction
pyrt-dophot -U '@.p2' images.fits

# Including pointing variation
pyrt-dophot -y -U '@.p2' images*.fits

# Add color correction
pyrt-dophot -y -U '@.p2,PC' images*.fits
```

### For Experienced Users
```bash
# Multi-night survey data
pyrt-dophot -y -U '@.p3,&.r2,PC,PD' night*/*.fits

# Known instrument calibration
pyrt-dophot -y -U '#PC=0.12,#PD=0.03,@.p2' science*.fits

# Changing aperture conditions (frost/dew on primary mirror)
pyrt-dophot -y -U '@.p2,&*P2R' night_long_sequence/*.fits
```

### For Advanced Calibration
```bash
# Full polynomial exploration
pyrt-dophot -y -U '@.p4,@.r3,@.c2,@.d2' calibration/*.fits

# Conservative but thorough
pyrt-dophot -y -U '&.p2,&.r2,&PC,@.p3' survey_data/*.fits

# Fixed from previous runs
pyrt-dophot -U '#PC=0.123,#PD=0.045,#PR=-0.02,@.p2' followup*.fits
```

## The Philosophy: Making Complexity Simple

Photometric calibration is inherently complex - you're solving for dozens of parameters simultaneously across multiple images with varying conditions. But that doesn't mean the **interface** has to be complex.

The `-U` syntax follows a simple principle: **write what you mean**.

### Inspired by TPoint

This term syntax approach was inspired by **TPoint** by Patrick Wallace - the legendary telescope pointing modeling software. TPoint's elegant term language for fitting complex pointing models showed how powerful and intuitive a well-designed syntax can be. Just as TPoint makes telescope pointing corrections accessible through clear term notation, dophot3's `-U` syntax makes photometric calibration straightforward.

The parallel is beautiful: TPoint fits for mechanical pointing errors using terms like `IH`, `ID`, `CH`, etc., while pyrt-dophot fits for photometric systematics using terms like `PX`, `PC`, `PR`, etc. Both use the same philosophy of **domain-specific vocabulary** that matches how experts think about the problem.

- Need spatial corrections? → `.p2`
- Want the computer to decide? → `@.p2`
- Know it's required? → `&.p2`
- Different for each image? → `&*.p2`
- Know the exact value? → `#PC=0.1`

Instead of forcing you to memorize dozens of separate options, the system provides a **vocabulary** that matches how astronomers naturally think about calibration problems.

The result? You spend more time doing science and less time fighting with software. 🔭✨

---

## The Unified System: One Pond, Many Lily Pads 🐸

*Teaching small frogs about how all the different ways of getting photometric terms work together harmoniously*

### The Universal Truth: All Sources Are Created Equal

pyrt-dophot has learned to treat **all sources of photometric information identically**. Whether your terms come from:

- **RESPONSE headers** (embedded in FITS files from previous runs)
- **Model files** (`-M model.ecsv`)
- **Command line assignments** (`-U '#PC=0.12'`)

...they all flow into the **same unified system** and get treated **exactly the same way**.

### The Three-Step Dance

Every pyrt-dophot run follows the same elegant choreography:

#### Step 1: Gather Initial Values from Everywhere
```bash
# From RESPONSE header: Z=23.56, PXY=0.14, P2X=0.32, P3X=-0.14
# From model file: PE=0.05, PAE=-0.09, PC=0.12
# From command line: -U '#PD=0.03,P2R=0.01'

# Result: One combined pool of initial values
combined_initial_values = {
    'Z': 23.56, 'PXY': 0.14, 'P2X': 0.32, 'P3X': -0.14,
    'PE': 0.05, 'PAE': -0.09, 'PC': 0.12, 'PD': 0.03, 'P2R': 0.01
}
```

#### Step 2: Source-Agnostic Pre-Selection
```bash
# Smart warm start: ANY term with initial values gets pre-selected
# Doesn't matter if it came from RESPONSE, model, or command line!

stepwise_candidates = ['P3Y', 'P4X', 'PR', 'P2R']
pre_selected_terms = ['PXY', 'P2X', 'P3X', 'PC', 'P2R']  # Have initial values

print("Pre-selected term: PXY = 0.14")  # From RESPONSE
print("Pre-selected term: PC = 0.12")   # From model
print("Pre-selected term: P2R = 0.01")  # From command line
```

#### Step 3: Unified Refinement
The stepwise regression doesn't know or care where terms came from - it just optimizes the best photometric model using all available information.

### Real-World Scenarios

#### Scenario 1: D50 Workflow Speed Boost 🚀
```bash
# First image: Cold start (5.2 seconds)
pyrt-dophot img001.fits -U '.p3,.r3'
# → Fits from scratch, saves RESPONSE header in FITS file

# Second image: Warm start (1.2 seconds)
pyrt-dophot img002.fits -U '.p3,.r3'
# → Loads RESPONSE automatically, pre-selects proven terms
# → 4x speedup from intelligent warm starting!
```

#### Scenario 2: Model Training Strategy 🧠
```bash
# Step 1: Train comprehensive model
pyrt-dophot training_set*.fits -y -U '@.p4,@.r3,@PC,@PD' -W survey_model.ecsv

# Step 2: Apply to individual images
pyrt-dophot target001.fits -M survey_model.ecsv -U '@.p2'
# → Loads survey_model terms as initial values
# → Adds .p2 as stepwise candidates
# → Gets best of both: proven complex model + local optimization
```

#### Scenario 3: Mixed Source Harmony 🎵
```bash
# Complex multi-source fitting
pyrt-dophot -M base_model.ecsv -U '#PC=0.123,@.p3' archive_imgs/*.fits
# → Model file provides: PE, PAE, color terms, radial corrections
# → Command line fixes: PC at precisely calibrated value
# → RESPONSE headers provide: per-image zeropoints and spatial terms
# → All sources merge seamlessly into one coherent model
```

### The Frog's Wisdom: No More Lily Pad Confusion!

**Old way** (before unification):
- RESPONSE headers → mysterious different behavior
- Model files → different initialization
- Command line → yet another pathway
- Small frogs got confused about which lily pad to jump to! 🐸❓

**New way** (unified system):
- **One pond, one system** → all sources flow into same initial values pool
- **Source-agnostic warm start** → any term with initial value gets pre-selected
- **Intelligent merging** → no conflicts, no duplications, no surprises
- **Frog-friendly** → always behaves exactly the same way! 🐸✨

### Advanced Frog Techniques

#### The Model Evolution Pattern
```bash
# Generation 1: Learn from data
pyrt-dophot survey_night1*.fits -U '@.p3,@.r2,@PC' -W night1_model.ecsv

# Generation 2: Build on knowledge
pyrt-dophot survey_night2*.fits -M night1_model.ecsv -U '@.p3,@PD' -W night2_model.ecsv
# → Inherits night1 terms + explores new ones

# Generation 3: Production ready
pyrt-dophot new_targets*.fits -M night2_model.ecsv
# → Fast, robust, proven calibration
```

#### The Response Chain Technique
```bash
# Each image builds on the last
pyrt-dophot img001.fits -U '.p3,.r3'           # → RESPONSE saved in img001
pyrt-dophot img002.fits -U '.p3,.r3'           # → Loads img001 RESPONSE
pyrt-dophot img003.fits -U '.p3,.r3,@PC'       # → Loads img002 + explores color
# → Natural evolution of photometric model through observation sequence
```

### The Philosophy: Unified Simplicity

The unified system embodies a simple truth: **the source shouldn't matter, only the science**.

Whether your photometric information comes from:
- Previous successful calibrations (RESPONSE)
- Carefully constructed reference models (model files)
- Precise laboratory measurements (command line values)

...it should all work together seamlessly to give you the **best possible photometric calibration**.

No more mental juggling of "which technique do I use for this?" - just **one unified approach** that intelligently combines all available knowledge.

### Small Frog Summary 🐸

*"The wise pond-master taught us: Whether knowledge comes from yesterday's lily pad jumps (RESPONSE), the elder frog's wisdom (model files), or today's fresh observations (command line), all wisdom flows into the same pond. The unified system remembers everything and helps every frog jump more accurately!"*

**Translation for astronomers**: Source-agnostic initialization + intelligent warm starting = faster, more robust photometric calibration that just works.

---

*"The best software is invisible - it does exactly what you expect and gets out of your way."*
