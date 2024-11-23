#!/usr/bin/env python3

from astroquery.gaia import Gaia
import numpy as np
from astropy.table import Table, vstack
import warnings
import time
from tqdm import tqdm
import sys
import healpy as hp

def get_healpix_query(nside, ipix):
    """
    Generate query for a specific HEALPix pixel.
    Uses RA/Dec bounds instead of polygon for better ADQL compatibility.
    """
    # Get the corners of the HEALPix pixel
    corners = hp.boundaries(nside, ipix, step=1)
    ra, dec = hp.rotator.vec2dir(corners, lonlat=True)

    # Get bounds
    ra_min, ra_max = np.min(ra), np.max(ra)
    dec_min, dec_max = np.min(dec), np.max(dec)

    # Handle RA wrap-around
    if ra_max - ra_min > 180:
        ra_condition = f"(ra <= {ra_max} OR ra >= {ra_min})"
    else:
        ra_condition = f"ra BETWEEN {ra_min} AND {ra_max}"

    query = f"""
    SELECT
        source_id, ra, dec,
        pmra, pmdec,
        phot_g_mean_mag, phot_g_mean_flux_over_error,
        phot_bp_mean_mag, phot_bp_mean_flux_over_error,
        phot_rp_mean_mag, phot_rp_mean_flux_over_error,
        parallax, parallax_over_error,
        ruwe
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag < 8.0
        AND ruwe < 1.4
        AND visibility_periods_used >= 8
        AND phot_g_mean_mag IS NOT NULL
        AND phot_bp_mean_mag IS NOT NULL
        AND phot_rp_mean_mag IS NOT NULL
        AND parallax IS NOT NULL
        AND phot_g_mean_flux_over_error > 0
        AND phot_bp_mean_flux_over_error > 0
        AND phot_rp_mean_flux_over_error > 0
        AND {ra_condition}
        AND dec BETWEEN {dec_min} AND {dec_max}
    """
    return query

def process_results(results):
    """
    Process query results into standardized format.
    """
    if len(results) == 0:
        return None

    output = Table()

    # Basic astrometry
    output['radeg'] = results['ra'].astype(np.float64)
    output['decdeg'] = results['dec'].astype(np.float64)

    # Handle proper motions, converting from mas/yr to deg/yr
    output['pmra'] = np.float32(results['pmra'].filled(0.0)) / (3.6e6)
    output['pmdec'] = np.float32(results['pmdec'].filled(0.0)) / (3.6e6)

    # Gaia magnitudes
    output['G'] = np.float32(results['phot_g_mean_mag'])
    output['BP'] = np.float32(results['phot_bp_mean_mag'])
    output['RP'] = np.float32(results['phot_rp_mean_mag'])

    # Calculate magnitude errors from flux_over_error
    output['G_err'] = np.float32(2.5 / (results['phot_g_mean_flux_over_error'] * np.log(10)))
    output['BP_err'] = np.float32(2.5 / (results['phot_bp_mean_flux_over_error'] * np.log(10)))
    output['RP_err'] = np.float32(2.5 / (results['phot_rp_mean_flux_over_error'] * np.log(10)))

    # Add parallax information
    output['parallax'] = np.float32(results['parallax'].filled(0.0))
    parallax_err = results['parallax'].filled(0.0) / results['parallax_over_error'].filled(1.0)
    output['parallax_err'] = np.float32(np.where(np.isfinite(parallax_err), parallax_err, 0.0))

    # Calculate Johnson magnitudes
    bpg = output['BP'] - output['G']
    grp = output['G'] - output['RP']

    # Johnson B
    output['Johnson_B'] = np.float32(output['BP'] + 0.0085
        + 0.666169*grp + 0.145798*grp*grp + 0.517584*grp*grp*grp)

    # Johnson V
    output['Johnson_V'] = np.float32(output['G'] + 0.0052
        + 0.368628*bpg + 0.139117*bpg*bpg
        - 0.103787*grp + 0.194751*grp*grp + 0.156161*grp*grp*grp)

    # Johnson R
    output['Johnson_R'] = np.float32(output['G'] + 0.0167
        - 0.036933*bpg - 0.083075*bpg*bpg + 0.022532*bpg*bpg*bpg
        - 0.353593*grp - 0.482768*grp*grp + 0.728803*grp*grp*grp)

    # Johnson I
    output['Johnson_I'] = np.float32(output['RP'] - 0.0660
        - 0.395266*bpg + 0.183776*bpg*bpg - 0.020945*bpg*bpg*bpg
        + 0.253449*grp - 0.113991*grp*grp)

    return output

def fetch_bright_stars():
    """
    Fetch all Gaia stars brighter than magnitude 8 using HEALPix tessellation.
    """
    print("Fetching bright stars from Gaia DR3...")

    # Use NSIDE=8 which gives 768 pixels, each covering ~53.7 square degrees
    NSIDE = 8
    npix = hp.nside2npix(NSIDE)
    print(f"Dividing sky into {npix} regions...")

    all_results = []
    errors = 0
    with tqdm(total=npix) as pbar:
        for ipix in range(npix):
            try:
                query = get_healpix_query(NSIDE, ipix)
                job = Gaia.launch_job(query)
                results = job.get_results()

                if len(results) > 0:
                    processed = process_results(results)
                    if processed is not None:
                        all_results.append(processed)

            except Exception as e:
                print(f"\nError in pixel {ipix}: {str(e)}", file=sys.stderr)
                errors += 1
                if errors > 10:
                    print("Too many errors, aborting...", file=sys.stderr)
                    break

            pbar.update(1)

    if not all_results:
        print("No results found!")
        return

    # Combine all results
    print("\nCombining results...")
    output = vstack(all_results)

    # Remove duplicates that might occur at region boundaries
    print("Removing duplicates...")
    output = unique_sources(output)

    # Sort by magnitude
    output.sort('G')

    # Add metadata
    output.meta['CATNAME'] = 'GAIA_BRIGHT'
    output.meta['RELEASE'] = 'DR3'
    output.meta['MAGLIMIT'] = 8.0
    output.meta['EPOCH'] = 2016.0
    output.meta['DATE'] = time.strftime('%Y-%m-%d', time.gmtime())
    output.meta['NCATS'] = len(output)
    output.meta['HEALPIX'] = f"NSIDE={NSIDE}"

    output['radeg'].unit = 'deg'
    output['decdeg'].unit = 'deg'
    output['pmra'].unit = 'deg/yr'
    output['pmdec'].unit = 'deg/yr'
    output['parallax'].unit = 'mas'
    output['parallax_err'].unit = 'mas'

    # Define units for magnitude columns
    for col in ['G', 'BP', 'RP', 'G_err', 'BP_err', 'RP_err',
                'Johnson_B', 'Johnson_V', 'Johnson_R', 'Johnson_I']:
        output[col].unit = 'mag'

    # Save to FITS file
    outfile = 'gaia_bright_stars.fits'
    output.write(outfile, format='fits', overwrite=True)

    print(f"Saved {len(output)} stars to {outfile}")

def unique_sources(table, tolerance=1.0/3600.0):  # tolerance of 1 arcsec
    """
    Remove duplicate sources based on position.
    """
    # Convert to numpy arrays for faster computation
    ra = table['radeg'].data.astype(np.float64)
    dec = table['decdeg'].data.astype(np.float64)
    keep = np.ones(len(table), dtype=bool)

    for i in range(len(table)-1):
        if keep[i]:
            # Calculate angular distances to all subsequent stars
            dra = (ra[i+1:] - ra[i]) * np.cos(np.radians(dec[i]))
            ddec = dec[i+1:] - dec[i]
            dist = np.sqrt(dra*dra + ddec*ddec)

            # Mark close stars as duplicates
            close = np.where(dist < tolerance)[0]
            keep[i+1:][close] = False

    return table[keep]

if __name__ == "__main__":
    fetch_bright_stars()
