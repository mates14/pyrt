# astrometry_refit.py

import numpy as np
#import astropy.wcs
#import astropy.io.fits
import zpnfit
import logging

def refine_fit(zpntest, data):
    """
    Refine the astrometric fit while maintaining both photometric and astrometric masks.
    """
    # Remove all masks to compute residuals for all points
    data.use_mask('default')
    adata_all = data.get_arrays('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
    residuals = zpntest.residuals(zpntest.fitvalues, adata_all)

    # Create the astrometric mask
    astro_mask = residuals < 3.0 * zpntest.wssrndf

    # Combine the photometric and astrometric masks
    data.use_mask('photometry')
    photo_mask = data.get_current_mask()
    combined_mask = photo_mask & astro_mask

    # Apply the combined mask
    data.add_mask('combined', combined_mask)
    data.use_mask('combined')

    # Refine the fit with the combined mask
    adata_ok = data.get_arrays('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
    zpntest.delin = True
    zpntest.fit(adata_ok)

def refit_astrometry(det, data, options):
    """
    Refit the astrometric solution for a given image

    Parameters:
    det : astropy.table.Table
        Detection table containing image metadata
    ra, dec : numpy.ndarray
        Right ascension and declination of matched stars
    image_x, image_y : numpy.ndarray
        X and Y pixel coordinates of matched stars
    image_dxy : numpy.ndarray
        Uncertainty in pixel coordinates
    options : argparse.Namespace
        Command line options

    Returns:
    astropy.wcs.WCS
        Updated WCS object
    numpy.ndarray
        Boolean array indicating which stars were used in the final fit
    """

    try:
        camera = det.meta['CCD_NAME']
    except KeyError:
        camera = "C0"

    if options.szp:
        zpntest = zpnfit.zpnfit(proj="AZP")
        zpntest.fitterm(["PV2_1"], [1])
#        zpntest.fitterm(["PV2_2"], [1e-6])

    # Initialize ZPN fit object based on camera type
    elif camera in ["C1", "C2", "makak", "makak2", "NF4", "ASM1"]:
        zpntest = zpnfit.zpnfit(proj="ZPN")
        zpntest.fixterm(["PV2_1"], [1])
    else:
        zpntest = zpnfit.zpnfit(proj="TAN")

    # Set up initial WCS parameters
    keys_invalid = setup_initial_wcs(zpntest, det.meta)

    if keys_invalid:
        logging.warning("I do not understand the WCS to be fitted, skipping...")
        return None

    # Set up camera-specific parameters
    setup_camera_params(zpntest, camera, options.refit_zpn)

    data.use_mask('photometry')

    # Perform the initial fit
    adata = data.get_arrays('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
    zpntest.fit(adata)

    # Refine the fit
    refine_fit(zpntest, data)
    refine_fit(zpntest, data)

    if options.sip is not None:
        zpntest.fixall()
        zpntest.add_sip_terms(options.sip)
        data.use_mask('photometry')
        refine_fit(zpntest, data)
        refine_fit(zpntest, data)

    # Save the model and print results
    zpntest.savemodel("astmodel.ecsv")
    print(zpntest)

    return zpntest

def setup_initial_wcs(zpntest, meta):
    """Set up initial WCS parameters."""
    keys_invalid = False
    for term in ["CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"]:
        try:
            zpntest.fitterm([term], [meta[term]])
        except KeyError:
            keys_invalid = True

    if keys_invalid:
        try:
            # Try to interpret old-fashioned WCS with CROTA
            zpntest.fitterm(['CD1_1'], [meta['CDELT1'] * np.cos(meta['CROTA1']*np.pi/180)])
            zpntest.fitterm(['CD1_2'], [meta['CDELT1'] * np.sin(meta['CROTA1']*np.pi/180)])
            zpntest.fitterm(['CD2_1'], [meta['CDELT2'] * -np.sin(meta['CROTA2']*np.pi/180)])
            zpntest.fitterm(['CD2_2'], [meta['CDELT2'] * np.cos(meta['CROTA2']*np.pi/180)])
            keys_invalid = False
        except KeyError:
            keys_invalid = True

    return keys_invalid

def setup_camera_params(zpntest, camera, refit_zpn):
    """Set up camera-specific parameters."""
    if camera == "C1":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [7.5, 386.1])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [2090,2043])
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [7.5, 386.1])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], [2090,2043])

    if camera == "C2":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [8.255, 343.8])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [2124.0,2039.0])
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [8.255, 343.8])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], [2124.0,2039.0])

    if camera in ( "makak2",  "makak"):
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [0.131823, 0.282538])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [813.6,622.8])
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [0.131823, 0.282538])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], [813.6,622.8])

    if camera == "NF4":
        zpntest.fitterm(["PV2_3"], [65.913305900171])
        zpntest.fitterm(["CRPIX1", "CRPIX2"], [522.75,569.96])

    if camera == "ASM1":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5", "PV2_5"], [-0.0388566,0.001255,-0.002769])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [2054.5,2059.0])
        else:
            zpntest.fixterm(["PV2_3", "PV2_5", "PV2_5"], [-0.0388566,0.001255,-0.002769])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], [2054.5,2059.0])
    # ... (similar blocks for other camera types)

