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

Now that you understand the basics, let's explore the exciting new features...

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