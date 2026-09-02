#!/usr/bin/env python3
"""
pyrt-galaxy-phcat: photometry pipeline for images with a host galaxy.

Workflow:
  0. If {base}h.fits does not already exist, prepare it automatically:
       a. Locate the PS1 master template ~/ps1-templates/{target}-{filter}.fits
          (TARGET/OBJECT and FILTER keywords from the science image header).
       b. Reproject the master to the science frame WCS with pyrt-combine.
       c. Run hotpants to produce {base}h.fits.
  1. Run pyrt-phcat on the original image to build the reference catalog.
  2. Capture the auto-selected FWHM and APERTURE from that run.
  3. Run pyrt-phcat on the hotpants-subtracted image ({base}h.fits) using
     the same FWHM/APERTURE so the photometry is on a consistent system.
  4. Locate the target (ORIRA, ORIDEC from the FITS header) in the
     hotpants catalog.
  5. Abort if another detection is found within 10 pixels of the target
     (dirty subtraction residual).
  6. Splice the target row (NUMBER=0) from the hotpants catalog into the
     original catalog, replacing the galaxy-contaminated placeholder.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import argparse

import numpy as np
import astropy.io.fits
import astropy.wcs
import astropy.table

from pyrt.cli.phcat import process_photometry

PS1_TEMPLATE_DIR  = os.path.expanduser("~/ps1-templates")
HOTPANTS_FALLBACK = "/home/mates/src/hotpants-master/hotpants"


def _find_hotpants():
    hp = shutil.which("hotpants") or (
        HOTPANTS_FALLBACK if os.path.exists(HOTPANTS_FALLBACK) else None
    )
    return hp


# Background-noise multipliers used to derive hotpants thresholds from data,
# rather than hardcoding fixed ADU counts that assume a particular sky level.
#
# FLOOR_SIGMA_MULT sets the lower bound (il/tl): generous on purpose, it only
# needs to reject genuinely broken pixels (dead columns, readout garbage) --
# doesn't need tuning, doesn't affect fit stability.
#
# CEIL_SIGMA_MULT sets the general upper bound (iu/tu): the value a pixel can
# take and still be considered "real data" for the difference image. Also not
# fit-critical in practice (verified: leaving it anywhere from ~600x to
# ~2500x sigma made no difference once the kernel-fit ceiling was right).
#
# KERNEL_SIGMA_CANDIDATES sets the *kernel-fit* upper bound (-iuk/-tuk), which
# is what actually matters. PS1 templates have saturated stellar cores
# running into the hundreds of thousands to millions of ADU, far above
# general validity; letting those into hotpants' stamp/kernel fit makes the
# fit blow up globally (every stamp rejected with a nonsensical chi2, output
# comes out NaN/1e-30 almost everywhere) even though hotpants still exits 0.
#
# The same absolute ceiling is applied to *both* -iuk and -tuk (not each
# image's own sigma independently, scaled off the template's sigma) --
# empirically that combination is what converges. But which exact stars end
# up inside vs. outside a given ceiling is a discrete, per-field accident of
# which star lands in which fit stamp, not a smooth function of the
# threshold: bisecting on a real failing frame found 3000 ADU gives a clean
# diffim (clipped stdev ~13 ADU) while 3500 gives one with clipped stdev
# ~500,000, with no monotonic trend in between. There is no single value
# that's safe in general -- this is the "art" part -- so instead of trusting
# one number, prepare_and_subtract() runs down this candidate list (as
# multiples of the template's own background sigma) and validates the actual
# output at each step (see _diffim_is_stable), keeping the first one that
# produces a sane difference image.
KERNEL_SIGMA_CANDIDATES = [50.0, 25.0, 75.0, 15.0, 100.0, 35.0, 10.0, 150.0]
FLOOR_SIGMA_MULT = 1000.0
CEIL_SIGMA_MULT  = 1000.0

# Approaches tried and abandoned while chasing this (2026-09-02), so they
# aren't re-explored from scratch next time a target still fails:
#
#  - Percentile-of-image instead of median+N*sigma for the kernel ceiling
#    (e.g. "exclude the top 1%"). No better: the same discrete instability
#    showed up across the 99.0-99.9 percentile range.
#  - Scaling -iuk off the *image's* own sigma and -tuk off the *template's*
#    own sigma independently, rather than one shared ceiling from the
#    template. Worse: the row-difference sigma of a resampled/coadded PS1
#    template isn't comparable to a raw science frame's, so independent
#    scaling either starved the science side of good calibration stars or
#    still let template spikes through.
#  - Widening the general validity ceiling (iu/tu) by a lot (thousands of
#    sigma) to be "safe". Didn't fix instability, and made hotpants dramatically
#    slower -- multi-minute runs, some effectively hanging -- so it's actively
#    counterproductive, not just unhelpful.
#  - Pedestal-shifting the data before calling hotpants (-ip/-tp) to move a
#    background-subtracted image's near-zero pixels away from zero, on the
#    theory that hotpants' Poisson noise model chokes on values near 0. Ruled
#    out: hotpants' noise calc uses fabs(), so negative pixels alone aren't
#    the problem, and pedestal-shifting made no difference in testing. Also
#    confirmed background-subtracted vs. sky-included science frames fail
#    identically -- the science image's background level isn't the cause.
#  - A single fixed multiplier (no retry). Doesn't exist: fit quality is not
#    a smooth function of the kernel ceiling for a given field (see below),
#    so no constant is safe across targets -- hence the candidate ladder.


def _background_stats(image_file, template_file):
    """(i_sigma, i_median, t_sigma, t_median) via the row-difference estimator."""
    from pyrt.cli.combine import calculate_background_stats

    i_sigma, i_median = calculate_background_stats(astropy.io.fits.getdata(image_file))
    t_sigma, t_median = calculate_background_stats(astropy.io.fits.getdata(template_file))
    print(f"  Background stats: image     median={i_median:.2f}  sigma={i_sigma:.2f}")
    print(f"                     template  median={t_median:.2f}  sigma={t_sigma:.2f}")
    return i_sigma, i_median, t_sigma, t_median


def _hotpants_thresholds(i_sigma, i_median, t_sigma, t_median, kernel_sigma_mult):
    """Build -il/-iu/-iuk/-tl/-tu/-tuk for one kernel_sigma_mult candidate."""
    kernel_ceil = t_median + kernel_sigma_mult * t_sigma
    return {
        "il":  i_median - FLOOR_SIGMA_MULT * i_sigma,
        "iu":  i_median + CEIL_SIGMA_MULT * i_sigma,
        "iuk": kernel_ceil,
        "tl":  t_median - FLOOR_SIGMA_MULT * t_sigma,
        "tu":  t_median + CEIL_SIGMA_MULT * t_sigma,
        "tuk": kernel_ceil,
    }


def _diffim_is_stable(diffim_file, sci_sigma, factor=200.0):
    """Sanity-check a hotpants difference image.

    A converged fit's residuals should stay within roughly the input noise
    level; a degenerate fit (near-singular kernel matrix, too few or badly
    chosen stamps) produces huge swings (1e4-1e6 ADU) even though hotpants
    still exits 0 and prints SUCCESS -- it doesn't detect its own failure.
    Compare a 1st/99th-percentile-clipped stdev against a generous multiple
    of the science image's own background sigma.
    """
    data = astropy.io.fits.getdata(diffim_file)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    lo, hi = np.percentile(finite, [1, 99])
    clipped_std = np.std(np.clip(finite, lo, hi))
    return clipped_std < factor * sci_sigma


def prepare_and_subtract(science_file, outfile, kernel_sigma_mult=None):
    """Reproject PS1 master template and run hotpants -> outfile.

    Reads TARGET (or OBJECT) and FILTER from the science image header to
    locate ~/ps1-templates/{target}-{filter}.fits.  The reprojected template
    is a temporary file that is removed when done.

    Returns True on success, False on any recoverable error.
    """
    hdr = astropy.io.fits.getheader(science_file)

    filt   = hdr.get("FILTER",  "").strip().removeprefix("Sloan-")
    target = f"{int(hdr['TARGET']):05d}" if "TARGET" in hdr else str(hdr.get("OBJECT", "")).strip().replace(" ", "_")

    if not filt:
        print("ERROR: FILTER keyword missing — cannot locate PS1 master", file=sys.stderr)
        return False
    if not target:
        print("ERROR: TARGET/OBJECT keyword missing — cannot locate PS1 master", file=sys.stderr)
        return False

    master = os.path.join(PS1_TEMPLATE_DIR, f"{target}-{filt}.fits")
    if not os.path.exists(master):
        orira  = hdr.get("ORIRA")
        oridec = hdr.get("ORIDEC")
        if orira is None or oridec is None:
            print(f"ERROR: PS1 master not found and ORIRA/ORIDEC missing: {master}", file=sys.stderr)
            return False
        print(f"PS1 master not found — downloading {master}")
        os.makedirs(PS1_TEMPLATE_DIR, exist_ok=True)
        from pyrt.cli.ps1mosaic import main as ps1mosaic_main
        ps1mosaic_main([str(orira), str(oridec), "30", filt, master])
        if not os.path.exists(master):
            print(f"ERROR: ps1mosaic failed to create {master}", file=sys.stderr)
            return False

    hotpants = _find_hotpants()
    if not hotpants:
        print(f"ERROR: hotpants not found in PATH or at {HOTPANTS_FALLBACK}", file=sys.stderr)
        return False

    combine_cmd = shutil.which("pyrt-combine") or f"{sys.executable} -m pyrt.cli.combine"

    # Reproject master into a temp dir so pyrt-combine's overwrite guard is happy
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/tmp")) as tmpdir:
        template_reproj = os.path.join(tmpdir, "ps1_template.fits")

        print(f"\nReprojecting PS1 master  {os.path.basename(master)}")
        print(f"  -> science frame WCS of {os.path.basename(science_file)}")
        result = subprocess.run(
            [*combine_cmd.split(),
             "--skel", science_file, "--no-selection",
             "-o", template_reproj, master],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: pyrt-combine failed:\n{result.stderr}", file=sys.stderr)
            return False

        i_sigma, i_median, t_sigma, t_median = _background_stats(science_file, template_reproj)

        # Try the user's/default candidate first, then fall back down the
        # standard ladder (skipping it there if it's a duplicate).
        candidates = [kernel_sigma_mult] if kernel_sigma_mult is not None else []
        candidates += [c for c in KERNEL_SIGMA_CANDIDATES if c not in candidates]

        for attempt, mult in enumerate(candidates, 1):
            thresh = _hotpants_thresholds(i_sigma, i_median, t_sigma, t_median, mult)
            print(f"Running hotpants (attempt {attempt}/{len(candidates)}, kernel-sigma-mult={mult:g})"
                  f"  {os.path.basename(science_file)} → {os.path.basename(outfile)}")
            print(f"  thresholds: il={thresh['il']:.0f} iu={thresh['iu']:.0f} iuk={thresh['iuk']:.0f}"
                  f"  tl={thresh['tl']:.0f} tu={thresh['tu']:.0f} tuk={thresh['tuk']:.0f}")
            result = subprocess.run([
                hotpants,
                "-il", f"{thresh['il']:.3f}", "-iu", f"{thresh['iu']:.3f}", "-iuk", f"{thresh['iuk']:.3f}",
                "-c", "t", "-n", "i",
                "-tl", f"{thresh['tl']:.3f}", "-tu", f"{thresh['tu']:.3f}", "-tuk", f"{thresh['tuk']:.3f}",
                "-inim",   science_file,
                "-tmplim", template_reproj,
                "-outim",  outfile,
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  hotpants exited with an error — trying next candidate:\n{result.stderr}",
                      file=sys.stderr)
                continue

            if _diffim_is_stable(outfile, i_sigma):
                print(f"  Difference image looks stable (kernel-sigma-mult={mult:g}).")
                break

            print("  Difference image looks unstable (degenerate kernel fit) — trying next candidate.")
        else:
            print(f"ERROR: hotpants did not converge to a stable fit after {len(candidates)} attempts "
                  f"(tried kernel-sigma-mult={candidates}).\n"
                  f"       Inspect {outfile} and/or try running hotpants by hand with different "
                  f"-iuk/-tuk.", file=sys.stderr)
            return False

    print(f"Subtracted image: {outfile}")
    return True


def read_options(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Photometry pipeline for images with a host galaxy.")
    parser.add_argument("-n", "--noiraf", action="store_true",
                        help="Do not use IRAF (pass through to phcat)")
    parser.add_argument("-a", "--aperture", type=float, default=None,
                        help="Force aperture for both phcat runs, overriding auto-selection")
    parser.add_argument("--max-target-dist", type=float, default=10.0,
                        help="Max pixel distance to accept as target detection (default: 10)")
    parser.add_argument("--kernel-sigma-mult", type=float, default=None,
                        help="How many background-sigma above sky a pixel may reach and still be "
                             "used for hotpants' kernel fit (-iuk/-tuk); tried first, before the "
                             f"automatic fallback ladder {KERNEL_SIGMA_CANDIDATES}")
    parser.add_argument("files", nargs="+", type=str,
                        help="Original (unsubtracted) FITS files to process")
    return parser.parse_args(args)


def run_one(file, noiraf=False, aperture_override=None, max_target_dist=10.0,
            kernel_sigma_mult=None):
    base = os.path.splitext(file)[0]
    hfile = base + "h.fits"

    if not os.path.exists(file):
        print(f"ERROR: {file} not found", file=sys.stderr)
        return False

    if not os.path.exists(hfile):
        print(f"No hotpants image found — running template preparation and subtraction")
        if not prepare_and_subtract(file, hfile, kernel_sigma_mult=kernel_sigma_mult):
            return False

    # ------------------------------------------------------------------ #
    # Step 1 – run phcat on the original image
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Step 1: phcat on original image: {file}")
    print(f"{'='*60}")
    tbl = process_photometry(file, noiraf=noiraf, aperture=aperture_override,
                            target_photometry=True)
    tbl.write(base + ".cat", format="ascii.ecsv", overwrite=True)
    print(f"Written: {base}.cat  ({len(tbl)} objects)")

    fwhm = tbl.meta["FWHM"]
    aperture = tbl.meta["APERTURE"]
    if aperture_override is not None:
        print(f"Captured: FWHM={fwhm:.3f}  APERTURE={aperture:.3f}  (aperture forced)")
    else:
        print(f"Captured: FWHM={fwhm:.3f}  APERTURE={aperture:.3f}")

    # ------------------------------------------------------------------ #
    # Step 3 – run phcat on the hotpants-subtracted image
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Step 3: phcat on hotpants image: {hfile}")
    print(f"        (using FWHM={fwhm:.3f}, APERTURE={aperture:.3f})")
    print(f"{'='*60}")
    htbl = process_photometry(hfile, noiraf=noiraf,
                              fwhm_override=fwhm, aperture=aperture,
                              target_photometry=True)
    htbl.write(base + "h.cat", format="ascii.ecsv", overwrite=True)
    print(f"Written: {base}h.cat  ({len(htbl)} objects)")

    # ------------------------------------------------------------------ #
    # Step 4 – locate target in hotpants catalog
    # ------------------------------------------------------------------ #
    hdr = astropy.io.fits.getheader(file)
    try:
        orira = hdr["ORIRA"]
        oridec = hdr["ORIDEC"]
    except KeyError as e:
        print(f"ERROR: missing header keyword {e}", file=sys.stderr)
        return False

    wcs = astropy.wcs.WCS(hdr)
    tx, ty = wcs.all_world2pix([orira], [oridec], 1)
    tx, ty = float(tx[0]), float(ty[0])
    print(f"\nTarget WCS position: RA={orira}  Dec={oridec}")
    print(f"Target pixel position: X={tx:.2f}  Y={ty:.2f}")

    dist = np.sqrt((htbl["X_IMAGE"] - tx) ** 2 + (htbl["Y_IMAGE"] - ty) ** 2)

    if len(dist) == 0:
        print("ERROR: hotpants catalog is empty", file=sys.stderr)
        return False

    nearest_idx = int(np.argmin(dist))
    nearest_dist = float(dist[nearest_idx])

    if nearest_dist > max_target_dist:
        print(f"ERROR: nearest hotpants detection is {nearest_dist:.1f} px away "
              f"(limit {max_target_dist:.0f} px) – target not detected?",
              file=sys.stderr)
        return False

    target_row = htbl[nearest_idx:nearest_idx + 1]  # keep as Table, not Row
    print(f"Found target  dist={nearest_dist:.2f} px  "
          f"MAG={float(target_row['MAG_AUTO'][0]):.3f} ± "
          f"{float(target_row['MAGERR_AUTO'][0]):.3f}")

    # ------------------------------------------------------------------ #
    # Step 5 – check for contaminating detections within 10 pixels
    # ------------------------------------------------------------------ #
    nearby_mask = dist <= max_target_dist
    n_nearby = int(np.sum(nearby_mask))
    if n_nearby > 1:
        print(f"\nWARNING: {n_nearby} detections within {max_target_dist:.0f} px "
              f"of target in hotpants catalog:", file=sys.stderr)
        for i in np.where(nearby_mask)[0]:
            row = htbl[i]
            print(f"  NUMBER={row['NUMBER']:4d}  "
                  f"X={float(row['X_IMAGE']):.1f}  Y={float(row['Y_IMAGE']):.1f}  "
                  f"dist={float(dist[i]):.1f} px  "
                  f"MAG={float(row['MAG_AUTO']):.3f}",
                  file=sys.stderr)
        # Prefer NUMBER=0 (injected placeholder) if present among nearby detections
        nearby_indices = np.where(nearby_mask)[0]
        zero_indices = [i for i in nearby_indices if int(htbl[i]["NUMBER"]) == 0]
        if zero_indices:
            nearest_idx = zero_indices[0]
            print("Using NUMBER=0 (injected entry) as target.", file=sys.stderr)
        else:
            print(f"Using nearest detection (NUMBER={int(htbl[nearest_idx]['NUMBER'])}).",
                  file=sys.stderr)
        target_row = htbl[nearest_idx:nearest_idx + 1]
        print(f"Found target  dist={float(dist[nearest_idx]):.2f} px  "
              f"MAG={float(target_row['MAG_AUTO'][0]):.3f} ± "
              f"{float(target_row['MAGERR_AUTO'][0]):.3f}  (revised)")

    # ------------------------------------------------------------------ #
    # Step 6 – splice hotpants target row into original catalog as NUMBER=0
    # ------------------------------------------------------------------ #
    target_row = astropy.table.Table(target_row)  # ensure it's a proper Table
    target_row["NUMBER"] = np.int32(0)
    target_row.meta.clear()  # avoid vstack metadata merge warnings

    # Drop any original-image detections near the target (galaxy contamination)
    orig_dist = np.sqrt((tbl["X_IMAGE"] - tx) ** 2 + (tbl["Y_IMAGE"] - ty) ** 2)
    tbl_rest = tbl[orig_dist > max_target_dist]

    merged = astropy.table.vstack([target_row, tbl_rest],
                                  metadata_conflicts="silent")
    merged.meta = tbl.meta  # restore original image metadata

    merged.write(base + ".cat", format="ascii.ecsv", overwrite=True)
    print(f"\nFinal catalog written: {base}.cat  "
          f"({len(merged)} objects, target from hotpants image)")
    return True


def main():
    opts = read_options()
    ok = True
    for f in opts.files:
        if not run_one(f, noiraf=opts.noiraf, aperture_override=opts.aperture,
                   max_target_dist=opts.max_target_dist,
                   kernel_sigma_mult=opts.kernel_sigma_mult):
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
