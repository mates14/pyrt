#!/usr/bin/python3
"""Download the General Catalogue of Variable Stars (GCVS / OKPZ) from
VizieR and cache it locally as a FITS table for offline use by f2cj.py's
--annotate feature. Re-run occasionally to refresh (GCVS is updated by
VizieR periodically, not in real time).
"""

import argparse
import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u


def fetch_gcvs():
    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1
    result = Vizier.query_constraints(catalog='B/gcvs/gcvs_cat')
    if not result:
        raise RuntimeError('VizieR query for B/gcvs/gcvs_cat returned no tables')
    table = result[0]

    # Entries with no coordinates are withdrawn/renamed/duplicate designations.
    has_coord = np.array([x.strip() != '' for x in table['RAJ2000']])
    table = table[has_coord]

    ra = Angle(table['RAJ2000'], unit=u.hourangle).deg
    dec = Angle(table['DEJ2000'], unit=u.deg).deg
    name = np.array([s.strip() for s in table['GCVS']])
    vartype = np.array([str(s) for s in table['VarType']])
    magmax = np.array(table['magMax'], dtype=float)
    magmax = np.where(np.isnan(magmax), -99.0, magmax)
    period = np.array(table['Period'], dtype=float)
    period = np.where(np.isnan(period), -1.0, period)

    cols = [
        fits.Column(name='name', format='16A', array=name),
        fits.Column(name='ra', format='D', unit='deg', array=ra),
        fits.Column(name='dec', format='D', unit='deg', array=dec),
        fits.Column(name='vartype', format='16A', array=vartype),
        fits.Column(name='magmax', format='E', unit='mag', array=magmax),
        fits.Column(name='period', format='E', unit='d', array=period),
    ]
    return fits.BinTableHDU.from_columns(cols), len(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalog_dir', help='Directory to write gcvs.fits into '
                        '(the same --catalog-dir passed to f2cj.py -a)')
    args = parser.parse_args()

    catalog_dir = os.path.expanduser(args.catalog_dir)
    os.makedirs(catalog_dir, exist_ok=True)
    out_path = os.path.join(catalog_dir, 'gcvs.fits')

    print('Querying VizieR for B/gcvs/gcvs_cat...')
    hdu, n = fetch_gcvs()
    hdu.writeto(out_path, overwrite=True)
    print(f'Wrote {n} variable stars to {out_path}')


if __name__ == '__main__':
    main()
