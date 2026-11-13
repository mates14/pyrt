#!/usr/bin/env python3
"""pyrt-deepfield: transient search in one deep image (a coadd) - pyrt.core.deepfield

  pyrt-deepfield deep.ecsv --frames 2*-dft.fits

deep.ecsv: the coadd's calibrated catalogue (pyrt-combine -> pyrt-phcat -> pyrt-cat2det ->
pyrt-dophot); --frames: the frames it was combined from, for the one-frame artefact check.
Writes deepfield.ecsv (the candidates, best first) and deepfield.reg, prints the
*** DEEPFIELD *** block."""

import argparse
import os
import sys
import warnings

import numpy as np
import astropy.table


def main(argv=None):
    ap = argparse.ArgumentParser(description="Transient search in one deep image (a coadd)")
    ap.add_argument('deep', help="calibrated catalogue of the deep image (ECSV)")
    ap.add_argument('--frames', nargs='+', default=[], help="the frames the deep image was combined from")
    ap.add_argument('--image', help="the deep FITS image (default: the catalogue's name with .fits)")
    ap.add_argument('--diff', help="its difference against a template (FITS, the deep image's grid)")
    ap.add_argument('--template', help="the Pan-STARRS master for the 'is anything there' test (default: "
                    "~/ps1-templates/{target}-{band}.fits when it exists)")
    ap.add_argument('--subtract', action='store_true',
                    help="make the difference against the Pan-STARRS template (pyrt-galaxy-phcat's subtraction) -> *-diff.fits")
    ap.add_argument('--no-online', action='store_true', help="no Pan-STARRS/Gaia query: every object is a candidate")
    ap.add_argument('--known', help="known-objects cache (default deepfield-known.ecsv next to the catalogue)")
    ap.add_argument('-o', '--output', default='deepfield.ecsv')
    ap.add_argument('-n', '--top', type=int, default=10, help="candidates printed")
    ap.add_argument('-v', '--verbose', action='store_true')
    o = ap.parse_args(argv)

    from pyrt.core import deepfield
    from pyrt.utils import grbinfo
    warnings.simplefilter('ignore')
    deep = astropy.table.Table.read(o.deep, format='ascii.ecsv')
    try:
        trig = grbinfo.get_trigger(None, meta=deep.meta)
    except Exception as e:
        print(f"deepfield: no trigger info ({e})")
        trig = None
    if trig is not None and trig.get('t0') is None:
        trig = None
    known = None
    if not o.no_online:
        ra, de = np.asarray(deep['ALPHA_J2000'], float), np.asarray(deep['DELTA_J2000'], float)
        ra0 = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(ra))), np.mean(np.cos(np.radians(ra)))))) % 360
        de0 = float(np.mean(de))
        rad = float(np.max(np.hypot(((ra - ra0 + 180) % 360 - 180) * np.cos(np.radians(de0)), de - de0))) * 1.05 + 0.005
        cache = o.known or os.path.join(os.path.dirname(os.path.abspath(o.deep)), 'deepfield-known.ecsv')
        known = deepfield.known_sources(ra0, de0, rad, cache=cache, verbose=o.verbose)
        if known is None:
            print("deepfield: WARNING no known-object catalogue: every object is a candidate")
    image = o.image or os.path.splitext(o.deep)[0] + '.fits'
    if not os.path.exists(image):
        print(f"deepfield: WARNING {image} not found: no sharpness test, no subtraction")
        image = None
    diff = o.diff
    if o.subtract and image is not None and diff is None:
        from pyrt.cli.galaxy_phcat import prepare_and_subtract
        diff = os.path.splitext(image)[0] + '-diff.fits'
        if not prepare_and_subtract(image, diff):
            print("deepfield: WARNING the subtraction failed, no difference-image test")
            diff = None
    template = o.template
    if template is None:
        band = str(deep.meta.get('FILTER', '')).replace('Sloan_', '').replace('Sloan-', '').strip()
        tgt = deep.meta.get('TARGET')
        cand_t = os.path.expanduser(f"~/ps1-templates/{int(tgt):05d}-{band}.fits") if tgt is not None and band else None
        template = cand_t if cand_t and os.path.exists(cand_t) else None
    if o.verbose:
        print(f"deepfield: Pan-STARRS master {template}" if template else "deepfield: no Pan-STARRS master")
    res = deepfield.deep_search(deep, frames=o.frames, trigger=trig, known=known, image=image, diff=diff,
                                template=template, verbose=o.verbose)
    res.write(o.output, format='ascii.ecsv', overwrite=True)
    colour = {'NEW': 'green', 'VARIABLE': 'yellow', 'ONEFRAME': 'red', 'FAINT': 'white', 'EXTENDED': 'cyan',
              'UNCHECKED': 'magenta', 'SHARP': 'red', 'IN_TEMPLATE': 'blue', 'DEFECT': 'red'}
    with open(os.path.splitext(o.output)[0] + '.reg', 'w') as f:
        f.write("# Region file format: DS9 version 4.1\nfk5\n")
        if trig is not None and np.isfinite(trig.get('ra', np.nan)):
            f.write('circle(%.6f,%.6f,%.2f") # color=cyan dash=1 text={%s}\n'
                    % (trig['ra'], trig['dec'], trig['errorbox'] * 3600, trig.get('name', '')))
        for r in res:
            f.write('circle(%.6f,%.6f,4") # color=%s text={%s %.2f}\n' % (r['RA'], r['DEC'], colour[r['CLASS']],
                                                                       r['CLASS'], r['MAG']))
    print(deepfield.report(res, deep, trig, top=o.top, verbose=o.verbose))
    return 0


if __name__ == '__main__':
    sys.exit(main())
