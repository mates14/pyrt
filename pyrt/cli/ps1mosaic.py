#!/usr/bin/env python3
"""
Download a PanSTARRS stack reference image for a given sky position, mosaicking
across skycell boundaries so the result always covers the full requested field.

PS1 sky is divided into overlapping ~0.4-deg skycells; a single fitscut request
clips silently at the cell boundary and pads the remainder with zeros.  This
script queries the 4 corners of the FoV as well as the centre to discover every
overlapping cell, downloads a cutout from each, and reprojects+coadds them into
one clean FITS file ready for image subtraction.
"""

import argparse
import io
import sys

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

PS1_FILENAMES_URL = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_CUTOUT_URL    = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
PS1_PIXSCALE      = 0.25   # arcsec/pixel
PS1_FILTERS       = "grizy"


def _query_filenames(ra, dec, filt):
    """Return stack filenames for the skycell containing (ra, dec)."""
    params = {"ra": ra, "dec": dec, "size": 0.01, "filters": filt, "type": "stack"}
    r = requests.get(PS1_FILENAMES_URL, params=params, timeout=30)
    r.raise_for_status()
    files = []
    for line in r.text.strip().splitlines():
        cols = line.split()
        # columns: projcell subcell ra dec filter mjd type filename shortname badflag
        if len(cols) < 8 or not cols[0].isdigit():
            continue
        files.append(cols[7])
    return files


def find_skycells(ra, dec, width_arcmin, filt="i"):
    """Return filenames of all PS1 stack skycells that overlap the FoV.

    Queries center + 4 corners so a cell that only touches the edge is found
    even when the center lies fully inside a single cell.
    """
    half = (width_arcmin / 60.0) / 2.0
    dra  = half / np.cos(np.radians(dec))

    query_points = [
        (ra,        dec       ),
        (ra + dra,  dec + half),
        (ra + dra,  dec - half),
        (ra - dra,  dec + half),
        (ra - dra,  dec - half),
    ]

    seen, files = set(), []
    for qra, qdec in query_points:
        for f in _query_filenames(qra, qdec, filt):
            if f not in seen:
                seen.add(f)
                files.append(f)
    return files


def download_cutout(filename, ra, dec, size_pix):
    """Download a FITS cutout centred on (ra, dec) from one skycell.

    Returns an HDUList, or None if the server returns an empty image.
    """
    params = {
        "red": "/" + filename.lstrip("/"),
        "format": "fits",
        "x": ra,
        "y": dec,
        "size": size_pix,
        "wcs": 1,
        "imagename": "cutout.fits",
    }
    r = requests.get(PS1_CUTOUT_URL, params=params, timeout=60)
    r.raise_for_status()
    if len(r.content) < 2880:   # less than one FITS block — empty response
        return None
    return fits.open(io.BytesIO(r.content))


def mosaic(cutouts):
    """Reproject all cutouts to a common WCS and mean-coadd them.

    Returns a single PrimaryHDU.  Requires the `reproject` package.
    """
    try:
        from reproject import reproject_interp
        from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
    except ImportError:
        print("error: the 'reproject' package is required for multi-cell mosaicking.\n"
              "       pip install reproject", file=sys.stderr)
        sys.exit(1)

    hdus = []
    for hdul in cutouts:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                hdus.append(hdu)
                break

    if len(hdus) == 1:
        return hdus[0]

    wcs_out, shape_out = find_optimal_celestial_wcs(
        [(h.data, WCS(h.header)) for h in hdus]
    )
    array, _ = reproject_and_coadd(
        [(h.data, WCS(h.header)) for h in hdus],
        wcs_out, shape_out=shape_out,
        reproject_function=reproject_interp,
        combine_function="mean",
    )
    return fits.PrimaryHDU(data=array.astype(np.float32), header=wcs_out.to_header())


def build_parser():
    p = argparse.ArgumentParser(
        prog="ps1mosaic",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Filters available: g r i z y  (default: i)\n"
               "Output defaults to ps1_<filter>_<ra>_<dec>.fits",
    )
    p.add_argument("ra",    type=float, help="right ascension in decimal degrees (J2000)")
    p.add_argument("dec",   type=float, help="declination in decimal degrees (J2000)")
    p.add_argument("width", type=float, help="field width in arcminutes")
    p.add_argument("filter", nargs="?", default="i",
                   metavar="FILTER",
                   help="PS1 filter: g r i z y  (default: i)")
    p.add_argument("outfile", nargs="?", default=None,
                   metavar="OUTFILE",
                   help="output FITS filename (default: ps1_<filter>_<ra>_<dec>.fits)")
    return p


def main(args=None):
    parser = build_parser()
    ns = parser.parse_args(args)

    if ns.filter not in PS1_FILTERS:
        parser.error(f"unknown filter '{ns.filter}'; choose from: {PS1_FILTERS}")

    outfile  = ns.outfile or f"ps1_{ns.filter}_{ns.ra:.4f}_{ns.dec:.4f}.fits"
    size_pix = int(round(ns.width * 60 / PS1_PIXSCALE))

    print(f"Region:  RA={ns.ra}  Dec={ns.dec}  width={ns.width}'={size_pix}px  filter={ns.filter}")
    print(f"Output:  {outfile}")

    print("Querying skycell coverage...")
    cells = find_skycells(ns.ra, ns.dec, ns.width, ns.filter)
    if not cells:
        print("error: no PS1 stack files found for this position/filter.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(cells)} skycell(s) found:")
    for c in cells:
        print(f"    {c}")

    cutouts = []
    for cell in cells:
        label = cell.split("/")[-1]
        print(f"  Downloading {label} ...", end=" ", flush=True)
        hdul = download_cutout(cell, ns.ra, ns.dec, size_pix)
        if hdul is None:
            print("empty — skipping")
        else:
            img = next(h for h in hdul if h.data is not None)
            print(f"ok  {img.data.shape}")
            cutouts.append(hdul)

    if not cutouts:
        print("error: all cutouts were empty.", file=sys.stderr)
        sys.exit(1)

    if len(cutouts) == 1:
        hdu = next(h for h in cutouts[0] if h.data is not None)
        hdu.writeto(outfile, overwrite=True)
    else:
        print(f"Mosaicking {len(cutouts)} cutouts...")
        mosaic(cutouts).writeto(outfile, overwrite=True)

    print(f"Done: {outfile}")


if __name__ == "__main__":
    main()
