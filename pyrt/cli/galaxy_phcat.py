#!/usr/bin/env python3
"""
pyrt-galaxy-phcat: photometry pipeline for images with a host galaxy.

Workflow:
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
import sys
import argparse

import numpy as np
import astropy.io.fits
import astropy.wcs
import astropy.table

from pyrt.cli.phcat import process_photometry


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
        print(f"ERROR: hotpants image {hfile} not found", file=sys.stderr)
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
        print(f"\nERROR: {n_nearby} detections within {max_target_dist:.0f} px "
              f"of target in hotpants catalog:", file=sys.stderr)
        for i in np.where(nearby_mask)[0]:
            row = htbl[i]
            print(f"  NUMBER={row['NUMBER']:4d}  "
                  f"X={float(row['X_IMAGE']):.1f}  Y={float(row['Y_IMAGE']):.1f}  "
                  f"dist={float(dist[i]):.1f} px  "
                  f"MAG={float(row['MAG_AUTO']):.3f}",
                  file=sys.stderr)
        print("Subtraction may not be clean. Aborting.", file=sys.stderr)
        return False

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
