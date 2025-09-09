# astrometry_refit.py

import os
import numpy as np
#import astropy.wcs
#import astropy.io.fits
import zpnfit
import logging
import matplotlib.pyplot as plt

def plot_astrometric_residuals(zpntest, data, filename="astrometric_residuals.png", arrow_scale=2.5):
    """
    Create diagnostic plot showing astrometric residuals as arrows

    Args:
        zpntest: Fitted ZPN model
        data: PhotometryData object with current mask
        filename: Output plot filename
        arrow_scale: Arrow scaling factor for visibility
    """
    try:
        # Get data arrays using current mask
        ad = data.get_fitdata('image_x', 'image_y', 'ra', 'dec', 'image_dxy')

        # Calculate catalog positions in image coordinates using fitted WCS
        x_cat, y_cat = zpntest.model(zpntest.fitvalues, ad.astparams)

        # Calculate residual vectors
        dx = ad.image_x - x_cat  # Detection - Catalog
        dy = ad.image_y - y_cat
        residual_mag = np.sqrt(dx**2 + dy**2)

        # Debug information
        print(f"Astrometric residuals: min={np.min(residual_mag):.4f}, max={np.max(residual_mag):.4f}, mean={np.mean(residual_mag):.4f}")
        print(f"Scaled arrows: min={np.min(residual_mag*arrow_scale):.1f}, max={np.max(residual_mag*arrow_scale):.1f} pixels")
        print(f"Image dimensions: x=[{np.min(ad.image_x):.0f}, {np.max(ad.image_x):.0f}], y=[{np.min(ad.image_y):.0f}, {np.max(ad.image_y):.0f}]")

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot stars as blue dots
        ax.scatter(ad.image_x, ad.image_y, c='blue', s=20, alpha=0.6, label='Detected positions')

        # Get image dimensions automatically
        x_min, x_max = np.min(ad.image_x) - 50, np.max(ad.image_x) + 50
        y_min, y_max = np.min(ad.image_y) - 50, np.max(ad.image_y) + 50

        med_size = np.sqrt(np.power(np.median(np.abs(dy)),2)+np.power(np.median(np.abs(dx)),2)) # ~0.5 pixel
        img_size = np.sqrt(np.power(x_max - x_min,2)+np.power(y_max - y_min,2)) # 1024 pixel, the arrow typical scale requested is arrow_scale (in % of image size) so:
        scaling = arrow_scale * img_size / 100 / med_size  # i.e. ->  for 5% should be x_scaling=0.05*1024/med_size_x ->
        print(f"The arrow scaling factor if {scaling} ({arrow_scale},{img_size},{med_size})")

        # Plot residual arrows (scaled for visibility)
        # Use better quiver parameters for small residuals
        ax.quiver(ad.image_x, ad.image_y, dx * scaling, dy * scaling,
                 residual_mag, scale_units='xy', scale=1, angles='xy',
                 cmap='viridis', alpha=0.8, width=0.002, headwidth=3, headlength=4)

        # Add colorbar for residual magnitude
        cbar = plt.colorbar(ax.collections[-1], ax=ax)
        cbar.set_label('Astrometric residual (pixels)', fontsize=12)

        # Get image dimensions automatically
        x_min, x_max = np.min(ad.image_x) - 50, np.max(ad.image_x) + 50
        y_min, y_max = np.min(ad.image_y) - 50, np.max(ad.image_y) + 50

        # Set equal aspect ratio and limits based on actual data
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Labels and title
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title(f'Astrometric Residuals (arrows scaled ×{arrow_scale})\n'
                    f'RMS: {np.sqrt(np.mean(residual_mag**2)):.3f} pixels', fontsize=14)

        # Add statistics text
        stats_text = f'Stars: {len(residual_mag)}\n'
        stats_text += f'RMS: {np.sqrt(np.mean(residual_mag**2)):.3f} px\n'
        stats_text += f'Max: {np.max(residual_mag):.3f} px\n'
        stats_text += f'Median: {np.median(residual_mag):.3f} px'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Save plot
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"Astrometric residual plot saved to {filename}")
        print(f"Astrometric residual plot saved to {filename}")

    except Exception as e:
        logging.warning(f"Failed to create astrometric residual plot: {e}")

def refine_fit(zpntest, data):
    """
    Refine the astrometric fit while maintaining both photometric and astrometric masks.
    """
    # Remove all masks to compute residuals for all points
    data.use_mask('default')
    ad_all = data.get_fitdata('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
    residuals = zpntest.residuals(zpntest.fitvalues, ad_all.astparams)

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
    ad_ok = data.get_fitdata('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
    zpntest.delin = True
    zpntest.fit(ad_ok.astparams)

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

    logging.info(f"CAMERA is {camera}")
    if options.szp:
        zpntest = zpnfit.zpnfit(proj="AZP")
        zpntest.fitterm(["PV2_1"], [1])
#        zpntest.fitterm(["PV2_2"], [1e-6])

    # Initialize ZPN fit object based on camera type
    elif camera in ["C1", "C2", "makak", "makak2", "NF4", "ASM1", "ASM-S", "SROT1"]:
        logging.info(f"ZPN projectin activated")
        zpntest = zpnfit.zpnfit(proj="ZPN")
        zpntest.fixterm(["PV2_1"], [1])
    elif camera in ["CAM-ZEA"]:
        zpntest = zpnfit.zpnfit(proj="ZEA")
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
    ad = data.get_fitdata('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
    zpntest.fit(ad.astparams)

    # Refine the fit
    refine_fit(zpntest, data)
    refine_fit(zpntest, data)
    print(zpntest)

    if options.sip is not None:
        zpntest.fixall()
        data.use_mask('photometry')
        if options.sip > 2:
            ad = data.get_fitdata('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
            rms,num = zpntest.fit_sip2(ad.astparams, options.sip)
            logging.info(f"Fitted a SIP{options.sip} on {num} objects with rms={rms}")
        else:
            zpntest.add_sip_terms(options.sip)
            refine_fit(zpntest, data)
            refine_fit(zpntest, data)

    # Save the model and print results
    zpntest.savemodel("astmodel.ecsv")
    print(zpntest)

    # Create diagnostic plot of astrometric residuals if plotting is enabled
    if options.plot:
        base_filename = os.path.splitext(det.meta['FITSFILE'])[0]
        plot_filename = f"{base_filename}-ast.png"
        plot_astrometric_residuals(zpntest, data, plot_filename, arrow_scale=2.5)

    # Error model analysis removed - now done in transients.py with full catalog

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
        try: # CROTA1 & 2 are optional, may be left to default=0 in stacked images
            crota1 = meta['CROTA1']*np.pi/180
        except:
            crota1 = 0
        try:
            crota2 = meta['CROTA2']*np.pi/180
        except:
            crota2 = 0
        try:
            # Try to interpret old-fashioned WCS with CROTA
            zpntest.fitterm(['CD1_1'], [meta['CDELT1'] * np.cos(crota1)])
            zpntest.fitterm(['CD1_2'], [meta['CDELT1'] * np.sin(crota1)])
            zpntest.fitterm(['CD2_1'], [meta['CDELT2'] * -np.sin(crota2)])
            zpntest.fitterm(['CD2_2'], [meta['CDELT2'] * np.cos(crota2)])
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

    if camera == "SROT1":
        if refit_zpn:
            logging.info(f"SROT1 setup being loaded (active)")
            zpntest.fitterm(["PV2_3", "PV2_5"], [38.561185, 3461.163423])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [1882.796706,2055.012734])
        else:
            logging.info(f"SROT1 setup being loaded (passive)")
            zpntest.fixterm(["PV2_3", "PV2_5"], [38.561185, 3461.163423])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], [1882.796706,2055.012734])

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
            zpntest.fitterm(["PV2_3", "PV2_5", "PV2_7"], [-0.0388566,0.001255,-0.002769])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [2054.5,2059.0])
        else:
            zpntest.fixterm(["PV2_3", "PV2_5", "PV2_7"], [-0.0388566,0.001255,-0.002769])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], [2054.5,2059.0])
    if camera == "ASM-S":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [-0.0456,0.0442])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], [128.5,128.5])
#            zpntest.fitterm(["CRPIX1", "CRPIX2"], [339.8,335.5])
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [-0.0456,0.0442])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], [128.5,128.5])
#            zpntest.fixterm(["CRPIX1", "CRPIX2"], [339.8,335.5])
# PV2_3   =  -0.045557711245 / ± 0.001318165008 (2.893396%)
# PV2_5   =   0.044189874766 / ± 0.002482740922 (5.618348%)
# CRPIX1  = 339.882095473812 / ± 0.262750191549 (0.077306%)
# CRPIX2  = 335.523756136020 / ± 0.325225858217 (0.096931%)

#    # ... (similar blocks for other camera types)

