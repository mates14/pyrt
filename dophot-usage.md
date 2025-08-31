# New dophot3 Features: A Simple Guide

## Getting Started: What is dophot3 and How Do I Use It?

Dear wise frog, before we explore the fancy new features, let me explain what dophot3 actually does and how to use it for the first time.

### The Basic Workflow

dophot3 is a **telescope-agnostic photometric calibration system** that transforms your raw astronomical images into properly calibrated photometry. Here's the complete workflow:

1. **Start with a FITS file** that has basic WCS (World Coordinate System) information
2. **Extract sources:** `phcat.py your_image.fits` → creates `your_image.cat` (requires SExtractor)
3. **Prepare detection table:** `cat2det.py your_image.cat` → creates `your_image.det` (merges source extraction with FITS metadata)
4. **Perform photometric calibration:** `dophot3.py your_image.det` → creates `your_image.ecsv`

The final `.ecsv` file contains everything: your original detections + calibrated magnitudes + improved WCS + all the metadata needed for science.

### Simplest Possible Start

```bash
# Most basic usage - just give it a FITS file with WCS:
dophot3.py your_image.fits

# The system will automatically run the full pipeline:
# your_image.fits → phcat.py → cat2det.py → dophot3.py → your_image.ecsv
```

### What You Get

The output `.ecsv` file is **self-contained and reusable**:
- Contains calibrated photometry for all detected sources
- Can be fed back to dophot3.py for iterative improvements
- Includes astrometric solutions and photometric models
- Ready for scientific analysis or transient detection

### Dependencies

- **Required:** SExtractor (for initial object detection and centroid positions)
- **Recommended:** IRAF (for precision photometry of detected objects - 100MB well invested)
- **Alternative:** `phcat.py -I` uses SExtractor-only mode (faster, lighter, but less trusted photometry)
- **Workflow:** SExtractor finds objects → IRAF measures them precisely (or SExtractor does both if using -I)
- **Telescope agnostic:** Works with any observatory (though you might encounter squeaks with unusual setups)

Now that you understand the basics, let's explore the fitting modes...

---

## Single Image vs Multi-Image Fitting: Choose Your Strategy

Our frog deserves to know that dophot3 can work in two distinct modes, each with its own strengths and purpose.

### Single Image Mode: Precision Photometry

**Perfect for:** Regular photometry of individual objects, building photometric databases

**How it works:**
```bash
# Basic single image fitting
dophot3.py your_image.fits -U '.p3,.r3'
```

This is the **D50 telescope workflow** - we actually run the code **twice**:
1. **First fit:** Primarily for astrometric solution refinement
2. **Second fit:** The definitive photometric calibration

**What gets fitted:**
- **Model terms** specified with `--terms` (`-U`) - your favorite `.p3,.r3` covers spatial distortions up to 3rd degree plus radial effects
- **Direct terms** marked with `&` (always included regardless of statistics)
- **Stepwise terms** marked with `@` (let the fit decide what's significant)
- **Default terms** (no prefix) follow the `--use-stepwise` parameter setting

**Frog-level explanation:** "When studying one lily pad in detail, you can measure everything about that specific pad very precisely, but you can't learn much about how lily pads differ from each other."

### Multi-Image Mode: Advanced Calibration

**Perfect for:** Color responses, airmass effects, complex flatfield modeling, survey work

**How it works:**
```bash
# Multiple images with per-image terms - synthetic flatfield with response tilt variation
dophot3.py image1.fits image2.fits image3.fits -U '.p3,.r3,*PX,*PY,SC'

# The above is so common it has a shortcut: --fit-xy (-y)
dophot3.py image1.fits image2.fits image3.fits -y -U '.p3,.r3,SC'

# Global zeropoint for survey work
dophot3.py survey*.fits -Z -U '.p3,.r3,PA'
```

**Key advantages:**
1. **Per-image terms** (marked with `*`) model effects that vary between exposures:
   - `*PX,*PY` = Response tilt variations per-image (so common it has `-y` shortcut)
   - `*P2R` = Radial distortion per-image (dew formation, thermal changes)
   - **Most common use:** Multi-image synthetic flatfield with slight response tilt correction

2. **Global vs per-image zeropoints:**
   - **Default:** Each image gets its own zeropoint (handles varying conditions)
   - **`--single-zeropoint` (`-Z`):** One global zeropoint + atmospheric modeling (PA terms)

3. **Better constraint of complex effects:**
   - **Color responses** need multiple filters across multiple images
   - **Airmass coefficients** need range of observing conditions
   - **Synthetic flatfield** expressions benefit from large sample sizes

**Frog-level explanation:** "When studying an entire pond, you can learn the universal rules that apply to all lily pads (like how they all react to temperature), plus measure how individual pads vary from the average (like which corner gets more sunlight)."

### Choosing the Right Mode

| Situation | Mode | Recommended Terms | Why |
|-----------|------|------------------|-----|
| **Single target photometry** | Single image | `.p3,.r3` | Focus on local precision |
| **Variable star monitoring** | Single image | `.p2,PC` | Consistent calibration per epoch |
| **Survey photometry** | Multi-image + `-Z` | `PA,.p3,.r3` | Global photometric system |
| **Color calibration** | Multi-image | `PC,PD,.p2` | Need multiple filters/conditions |
| **Challenging weather** | Multi-image + `-y` | `.p2,SC` | Per-image cloud corrections |

### The General Purpose Philosophy

dophot3 is a **general purpose tool** - the term combination and usage mode vary depending on your specific situation. The system provides the building blocks; you choose how to combine them based on your science goals and observing conditions.

**Important note:** While we speak of "modes," these are modes of **usage**, not different code paths. The photometry fitting code itself doesn't distinguish between single and multiple images - it simply fits all identified objects whether they come from one image or many. The only differences are marginal:

1. **Zeropoint naming:** Single image gets `Z`, multiple images get `Z:1`, `Z:2`, etc.
2. **Astrometric refitting:** Currently only implemented for single images (skipped for multi-image inputs)

### Advanced Workflow: Train Once, Apply Many

A powerful strategy combines both approaches: **use multi-image fitting to create a robust photometric model, then apply that model to individual images**.

```bash
# Step 1: Train the model using many images with good statistics
dophot3.py training_set*.fits -y -U '.p3,.r3,SC,PC' -W advanced_model.ecsv

# Step 2: Apply the trained model to single images
dophot3.py target_image.fits -M advanced_model.ecsv
```

**Why this works:**
- **Training phase:** Multi-image fit constrains complex terms that need large samples (color responses, high-order spatial terms)
- **Application phase:** Single images benefit from advanced corrections they couldn't determine alone
- **Computational efficiency:** A 1000-image super-fit would never finish - better to train on a selected representative group
- **Pipeline-ready:** The `-M` option supports filter-specific models for automated processing without overwhelming the system

**Filter-specific model logic:**
```bash
# If you specify a base model name:
dophot3.py image_g.fits -M /path/to/model

# The system tries:
# 1. /path/to/model (exact path)
# 2. /path/to/model-g.mod (filter-specific version)
```

This enables **catalog→physical color transformations** and other filter-dependent calibrations in automated pipelines.

**Real-world scenarios:**
- **Large surveys:** Instead of fitting 1000 images simultaneously (computationally impossible), select 20-50 representative images for model training
- **Pipeline operations:** Load pre-computed models to avoid complex fitting that could overwhelm automated systems
- **Time-critical processing:** Model loading finishes much faster than full fitting when rapid results are needed

**Frog-level explanation:** "You can't study every lily pad in the entire pond at once - that would take forever and probably crash your lily-pad-ometer! Instead, pick a good sample of lily pads to learn the rules, then apply those rules quickly to each individual pad you encounter."

Whether you're building a photometric database one image at a time or processing an entire night's survey data, the frog's wisdom applies: "Choose your pond size to match your lily pad research goals." 🐸

---

## What's New? Four Game-Changing Features

dophot3 has gained some powerful new capabilities that make astronomical photometry more robust and flexible. Let's explain them like we're talking to a wise pet frog who's really interested in astronomy.

---

## 1. Color Filtering: Keeping the Good Stars, Ditching the Weird Ones

### The Problem
Sometimes your star catalog contains objects that aren't actually normal stars. The worst offenders are **emission line objects** with impossible colors:

- **Planetary nebulae** - OIII emission in g-band, H-alpha in r-band, almost nothing in i/z → colors like g-r = -3!
- **Cataclysmic variables** - strong emission lines creating bizarre color combinations  
- **Active galaxies/quasars** - when dominated by emission lines rather than continuum
- **Artifacts** from cosmic rays or satellite trails

The problem isn't the object type itself, but **emission line contamination** that creates colors completely outside the stellar locus. A single object with g-r = -3 sits alone in "color space void" where the robust fitting can't ignore it - it forces the photometric model to bend toward these impossible colors.

**Why this breaks everything:** Normal galaxies and continuum quasars aren't too problematic, but emission line objects create colors so extreme they're immune to robust fitting rejection. When an object has g-r = -3, it sits in an empty region of color space where robust algorithms can't find "nearby" normal objects to help identify it as an outlier. The fitting algorithm has to accommodate this impossible color, distorting the entire photometric solution.

### The Solution: `-R` and `-B` Options
```bash
# Remove objects redder than i-z = +1.5
dophot3 -R 1.5 image.fits

# Remove objects bluer than g-r = -0.5  
dophot3 -B -0.5 image.fits

# Remove both extremes
dophot3 -R 1.5 -B -0.5 image.fits
```

**What happens under the hood:**
- dophot3 looks at all your star colors (g-r, r-i, i-z)
- It finds the **most extreme** color for each object (reddest of the reds, bluest of the blues)
- Objects beyond your limits get **excluded from the catalog entirely**
- They never even enter the fitting process

**Frog-level explanation:** "If one 'fly' is glowing bright purple (emission lines), it's so weird that even when you try to ignore strange flies, this one impossible fly forces you to think that all flies might be partially purple. Better to exclude the glowing fly entirely and catch the normal flies properly."

---

## 2. Target Exclusion: Don't Let Variable Stars Hijack Your Calibration

### The Problem  
You're studying a **variable star** like the OJ287 blazar. It's bright, and it's... completely ruining your photometric calibration! 

Why? Because dophot3 thinks it's a nice stable calibration star, but it's actually changing brightness. It's like trying to calibrate your fly-catching reflexes using a firefly that keeps blinking on and off.

### The Solution: `--exclude-target` (Default: ON)
```bash
# Target exclusion is ON by default - nothing to do!
dophot3 target_image.fits

# Turn it OFF if you really want to include the target
dophot3 --exclude-target=false target_image.fits
```

**What happens under the hood (RTS2-specific):**
- **RTS2** telescope control system stores the target's original database coordinates in **ORIRA/ORIDEC** header keywords
- dophot3 reads these coordinates and identifies the target object during image loading
- The target gets **excluded from catalog pairing** - it never enters the calibration process
- You still get photometry of the target (included in the output line), but it doesn't influence calibration
- Other objects are stored in the .ecsv file and can be retrieved by coordinates using `ecsv_target.py`

**Note:** This feature requires RTS2 header keywords. Other telescope systems can adopt this approach by storing target coordinates as ORIRA/ORIDEC in their FITS headers.

**Frog-level explanation:** "When measuring how fast normal flies buzz around, don't include that special firefly you're studying (the one that blinks in patterns) - you're here to study that firefly's blinking behavior, not use it to calibrate your 'normal fly speed' measurements."

---

## 3. Single Zeropoint Mode: One Calibration to Rule Them All

### The Problem
Normally, dophot3 fits a separate zeropoint for each image. This works great for small fields, but for **all-sky surveys** or **large mosaics**, you want a single photometric solution that works everywhere.

Think of it like pond temperature: instead of having different "what warm means" for each rock you sit on, you want one universal warmth scale for the entire pond.

### The Solution: `-Z` (Single Zeropoint)
```bash
# Fit one common zeropoint for all images
dophot3 -Z -U 'PAC,PC,P2C,&.p3,&.r3' image*.fits
```

**What happens under the hood:**
- Instead of fitting zeropoint₁, zeropoint₂, zeropoint₃, etc.
- dophot3 calculates one **average zeropoint** 
- Then it uses **airmass-dependent terms** (like PAC) to handle atmospheric extinction
- The result is a **unified photometric model** that works across your entire dataset

**Frog-level explanation:** "Instead of having different fly-counting methods for each lily pad, let's use the same fly-counting system across the whole pond, but account for the fact that flies near the surface are easier to see than flies deep underwater (atmospheric effects)."

---

## 4. Absolute Airmass Support: Better Atmospheric Modeling

### The Problem
Traditional photometry uses **relative airmass** - how much extra atmosphere you're looking through compared to straight up. But for large-field or all-sky work, you sometimes need **absolute airmass** - the total amount of atmosphere.

It's like the difference between "how much extra water am I looking through compared to surface level?" (relative) vs. "how much total water am I looking through?" (absolute).

### The Solution: A and B Terms
- **A terms** (like `PA`, `PAC`) use **absolute airmass**
- **B terms** (like `PB`) use **relative airmass** (traditional)

```bash
# Use absolute airmass for atmospheric reddening
dophot3 -U 'PAC,PC' image*.fits

# Use relative airmass for spatial gradients  
dophot3 -U 'PB,&.p2' image*.fits

# Mix both approaches
dophot3 -Z -U 'PAC,PB,PC,&.p3' allsky*.fits
```

**Key Applications:**

**PAC (Atmospheric Reddening):** Different star colors get absorbed differently by the atmosphere. Red light penetrates better than blue light (think sunset colors). PAC measures how much this **differential absorption** depends on how much atmosphere you're looking through.

**PB (Relative Airmass):** When making a **global flatfield** from multiple images of the same field at different airmasses, PB can model how brightness changes with atmospheric thickness while keeping the spatial patterns constant.

**Frog-level explanation:** "When studying how flies look different at dawn vs. dusk, you need to know both 'how much water is the light going through?' (absolute) and 'how much extra water compared to midday?' (relative). Different hunting strategies need different measurements."

---

## Putting It All Together: Real-World Examples

### Example 1: Survey Photometry with Problem Objects
```bash
# Remove extreme colors, exclude targets, use single zeropoint
dophot3 -B -0.5 -R 1.2 -Z -U 'PAC,PC,&.p2' survey*.fits
```
**Translation:** "Remove impossibly blue/red objects, don't let targets mess up calibration, use one zeropoint with atmospheric modeling and spatial corrections."

### Example 2: All-Sky Calibration
```bash  
# Ultimate all-sky photometric modeling
dophot3 -Z -U 'PAC,PC,P2C,P3C,&.p3,&.r3' allsky_data*.fits
```
**Translation:** "Single zeropoint for the whole sky, with atmospheric reddening, multiple color terms, and full spatial corrections."

### Example 3: Variable Star Field
```bash
# Study a variable star properly
dophot3 -B -0.3 --exclude-target=true -U 'PC,&.p2' blazar_images*.fits  
```
**Translation:** "Remove blue artifacts, definitely exclude the variable target from calibration, use basic color and spatial corrections."

---

## The Philosophy: Robust Photometry Made Simple

These features embody a simple principle: **photometry should be robust against real-world messiness**.

- **Color filtering** handles catalog contamination
- **Target exclusion** handles variable sources  
- **Single zeropoint** handles large-scale surveys
- **Absolute airmass** handles complex atmospheric effects

Each feature solves a specific problem that professional astronomers face daily, but the implementation is automatic and transparent.

**Frog-level summary:** "When hunting flies, account for the fact that some 'flies' might be glowing moths in disguise, some flies might be special fireflies you're studying, you might be hunting across multiple ponds with different lighting, and the water conditions might make flies look different colors depending on how deep you're looking."

The result? More reliable photometry with less croaking about problems. 🐸✨

---

*"The best calibration is the one you don't have to think about."*

---

*Note: The frog-themed explanations in this document honor the tradition of Ondřejov Observatory, inspired by Jan Neruda's "Písně kosmické" where an old frog explains the universe. The spirit of making cosmic mysteries accessible through simple wisdom has guided this observatory since its founding by the Frič brothers.*