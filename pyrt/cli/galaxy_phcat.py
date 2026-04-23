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


def prepare_and_subtract(science_file, outfile):
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

        print(f"Running hotpants  {os.path.basename(science_file)} → {os.path.basename(outfile)}")
        result = subprocess.run([
            hotpants,
            "-il", "-10000", "-iu", "10000",
            "-c", "t", "-n", "i",
            "-tl", "-10000", "-tu", "20000",
            "-inim",   science_file,
            "-tmplim", template_reproj,
            "-outim",  outfile,
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR: hotpants failed:\n{result.stderr}", file=sys.stderr)
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
    parser.add_argument("files", nargs="+", type=str,
                        help="Original (unsubtracted) FITS files to process")
    return parser.parse_args(args)


def run_one(file, noiraf=False, aperture_override=None, max_target_dist=10.0):
    base = os.path.splitext(file)[0]
    hfile = base + "h.fits"

    if not os.path.exists(file):
        print(f"ERROR: {file} not found", file=sys.stderr)
        return False

    if not os.path.exists(hfile):
        print(f"No hotpants image found — running template preparation and subtraction")
        if not prepare_and_subtract(file, hfile):
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

    # Drop any pre-existing NUMBER=0 placeholder from original catalog
    tbl_rest = tbl[tbl["NUMBER"] != 0]

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
                   max_target_dist=opts.max_target_dist):
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
