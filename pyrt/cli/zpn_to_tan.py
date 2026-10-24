#!/usr/bin/env python3

import os
import sys
import shutil
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from pyrt.core import zpnfit

def create_grid_points(naxis1, naxis2, ngrid=100):
    """Create grid points with denser sampling near edges and center"""
    # Create three regions with different density
    edge = ngrid // 4
    center = ngrid // 2
    
    # Create coordinate arrays with varying density
    x_edges = np.concatenate([
        np.linspace(0, naxis1*0.1, edge),
        np.linspace(naxis1*0.1, naxis1*0.9, center),
        np.linspace(naxis1*0.9, naxis1-1, edge)
    ])
    y_edges = np.concatenate([
        np.linspace(0, naxis2*0.1, edge),
        np.linspace(naxis2*0.1, naxis2*0.9, center),
        np.linspace(naxis2*0.9, naxis2-1, edge)
    ])
    
    xx, yy = np.meshgrid(x_edges, y_edges)
    return np.column_stack((xx.ravel(), yy.ravel()))

def calculate_fit_weights(pixels, crpix, naxis1, naxis2):
    """Calculate weights for fitting based on distance from center"""
    dx = pixels[:,0] - crpix[0]
    dy = pixels[:,1] - crpix[1]
    r2 = dx*dx + dy*dy
    max_r2 = max(naxis1*naxis1, naxis2*naxis2) / 4  # Normalize by quarter of diagonal
    
    # Weight drops off with radius but not too sharply
    weights = 1.0 / (1.0 + r2/max_r2)
    return weights

def fit_sip_in_pixel_space(zpn_wcs, tan_wcs, pixels, order=4):
    """Fit the SIP polynomial that takes the TAN solution onto the ZPN one.

    zpnfit keeps its SIP in the sky -> pixel direction (FITS AP_/BP_): the
    pixel offset from CRPIX is ideal + sip(ideal), ideal being CD^-1 of the
    TAN intermediate coordinates; zpnfit.sip_header() inverts it to A_/B_.

    Args:
        zpn_wcs: Original ZPN WCS
        tan_wcs: TAN WCS (already fitted)
        pixels: Nx2 array of 1-based (FITS) pixel coordinates
        order: Maximum SIP order to fit

    Returns:
        a_coeffs, b_coeffs, span: AP_/BP_ coefficient matrices and the largest
        ideal offset the fit covered
    """
    sky = zpn_wcs.all_pix2world(pixels, 1)
    tan_pixels = tan_wcs.wcs_world2pix(sky, 1)

    valid = np.all(np.isfinite(tan_pixels), axis=1)
    if not np.any(valid):
        raise RuntimeError("TAN world2pix failed for all mesh points during SIP fitting")
    pixels = pixels[valid]
    tan_pixels = tan_pixels[valid]

    # ideal offsets from CRPIX, and what the SIP has to add to them
    crpix = tan_wcs.wcs.crpix
    u = tan_pixels[:,0] - crpix[0]
    v = tan_pixels[:,1] - crpix[1]
    dx = pixels[:,0] - tan_pixels[:,0]
    dy = pixels[:,1] - tan_pixels[:,1]

    # scaled powers keep the design matrix well conditioned
    span = float(max(np.max(np.abs(u)), np.max(np.abs(v))))
    terms = [(p, q) for p in range(order + 1)
             for q in range(order + 1 - p) if p + q > 0]
    A = np.column_stack([(u/span)**p * (v/span)**q for p, q in terms])

    x_solution = np.linalg.lstsq(A, dx, rcond=None)[0]
    y_solution = np.linalg.lstsq(A, dy, rcond=None)[0]

    a_coeffs = np.zeros((order + 1, order + 1))
    b_coeffs = np.zeros((order + 1, order + 1))
    for (p, q), a, b in zip(terms, x_solution, y_solution):
        a_coeffs[p,q] = a / span**(p + q)
        b_coeffs[p,q] = b / span**(p + q)

    return a_coeffs, b_coeffs, span

def zpn_to_tan_mesh(header, ngrid=100, sip_order=4):
    """Convert ZPN to TAN using mesh fitting approach"""
    
    # Create WCS objects for input ZPN
    w_zpn = WCS(header)
    
    # Get image dimensions
    naxis1 = header['NAXIS1']
    naxis2 = header['NAXIS2']
    crpix = [header['CRPIX1'], header['CRPIX2']]
    
    # Create grid points, 1-based like CRPIX
    pixels = create_grid_points(naxis1, naxis2, ngrid) + 1
    
    # Calculate weights
    weights = calculate_fit_weights(pixels, crpix, naxis1, naxis2)
    
    # Project points through ZPN
    sky = w_zpn.all_pix2world(pixels, 1)
    
    # Initialize TAN fitter
    fitter = zpnfit.zpnfit(proj="TAN")
    
    # Fix the basic WCS parameters
    fitter.fixterm(["CRVAL1", "CRVAL2"], [header['CRVAL1'], header['CRVAL2']])
    fitter.fixterm(["CRPIX1", "CRPIX2"], [header['CRPIX1'], header['CRPIX2']])
    
    # Step 1: Fit just TAN projection (CD matrix)
    cd_terms = []
    cd_values = []
    for i in (1, 2):
        for j in (1, 2):
            key = f'CD{i}_{j}'
            if key in header:
                cd_terms.append(key)
                cd_values.append(header[key])

    # Header may use PC matrix + CDELT instead of CD matrix; derive equivalent
    if not cd_terms:
        cd = w_zpn.pixel_scale_matrix
        for i in range(2):
            for j in range(2):
                cd_terms.append(f'CD{i+1}_{j+1}')
                cd_values.append(float(cd[i, j]))

    fitter.fitterm(cd_terms, cd_values)
    
    # Prepare fit data with weights
    fit_data = (pixels[:,0],        # x pixel coordinates
               pixels[:,1],        # y pixel coordinates
               sky[:,0],          # RA
               sky[:,1],          # Dec
               weights)           # errors as weights
    
    #print("\nFitting TAN projection (CD matrix):")
    fitter.fit(fit_data)
    #print(fitter)
    w_tan = WCS(fitter.wcs())
    
    # Step 2: Fix TAN parameters and fit SIP terms
    fitter.fixall()
    fitter.add_sip_terms(sip_order)
   
    #print(f"\nFitting SIP terms (order {sip_order}):")
    a_coeffs, b_coeffs, span = fit_sip_in_pixel_space(w_zpn, w_tan, pixels, sip_order)

    # Update fitter with new SIP coefficients (AP_/BP_ direction)
    fitter.sip_a = a_coeffs
    fitter.sip_b = b_coeffs
    fitter.sip_order = sip_order
    fitter.sip_span = span
 
    w_new = WCS(fitter.wcs())
    
    # For validation, compute RMS in pixels (quiet=True returns NaN for diverging points)
    pixels_back = w_new.all_world2pix(sky, 1, quiet=True)
    valid = np.all(np.isfinite(pixels_back), axis=1)
    diff = pixels[valid] - pixels_back[valid]
    rms = np.sqrt(np.mean(weights[valid] * (diff[:,0]**2 + diff[:,1]**2)))

    # Print before/after residuals for sample points
    tan_pix = w_tan.all_world2pix(sky[:10], 0)  # first 10 points
    sip_dx, sip_dy = apply_sip(tan_pix[:,0], tan_pix[:,1], 
                              fitter.sip_a, fitter.sip_b, sip_order)
    #print("Sample point residuals:")
    #print("Original:", pixels[:10] - tan_pix)
    #print("With SIP:", pixels[:10] - (tan_pix + np.column_stack((sip_dx, sip_dy))))
    
    return fitter, rms

def apply_sip(x, y, a_coeffs, b_coeffs, order):
    """Apply SIP distortion explicitly"""
    dx = dy = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i + j > 0:
                xp = x**i * y**j
                dx += a_coeffs[i,j] * xp
                dy += b_coeffs[i,j] * xp
    return dx, dy

def main():
    """Convert ZPN FITS files to TAN projection with SIP corrections."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <fits_file>")
        sys.exit(1)

    fitsfile = sys.argv[1]
    outname = fitsfile.replace('.fits', '_tan.fits')

    # First copy the input file
    if os.path.exists(outname):
        os.unlink(outname)
    shutil.copy2(fitsfile, outname)

    # Read input file
    hdul = fits.open(fitsfile)
    header = hdul[0].header
    hdul.close()

    # Try different SIP orders
#    for order in [3]:
#        print(f"\n=== Testing SIP order {order} ===")
    fitter, rms = zpn_to_tan_mesh(header, ngrid=200, sip_order=5)
    print(f"\nRMS residual: {rms:.3f} pixels")
    #print(f"Final RMS: {rms:.3f} pixels")

    # Write best solution to file
    fitter.write(outname)


if __name__ == "__main__":
    main()
