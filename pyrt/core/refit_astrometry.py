# astrometry_refit.py

import os
import numpy as np
#import astropy.wcs
#import astropy.io.fits
from pyrt.core import zpnfit
import logging
import matplotlib.pyplot as plt
from copy import deepcopy

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

def select_best_projection(zpntest, data):
    """
    Test TAN, ZEA, AZP, and ZPN projections to find the best fit.

    Args:
        zpntest: Initial ZPN fitting object
        data: PhotometryData with astrometric data

    Returns:
        zpnfit object with best projection selected

    Projections tested:
        TAN - Gnomonic (standard, 0 parameters)
        ZEA - Zenithal equal area (photometry-friendly, 0 parameters)
        AZP - Zenithal perspective with fitted mu (projection type selection, 1 parameter)
              mu=0→TAN, mu=1→STG, mu→∞→SIN. No Runge oscillations like ZPN.
        ZPN - Zenithal polynomial (radial distortion, stepwise selection of polynomial terms)
              If no polynomial terms improve fit, reported as ARC (zenithal equidistant)
        ARC - Zenithal equidistant (r=u), widely used for Schmidt telescopes
              Mathematically equivalent to ZPN with only PV2_1=1
    """
    best_projection = None
    best_wssrndf = float('inf')
    best_zpntest = None
    best_terms = []

    # Test each projection type
    for proj_name in ["TAN", "ZEA", "AZP", "ZPN"]:
        logging.info(f"Testing {proj_name} projection...")
        print(f"Testing {proj_name} projection...")

        # Create a new zpnfit object with this projection
        zpntest_test = deepcopy(zpntest)
        proj_idx = zpntest_test.projections.index(proj_name)

        # Update the PROJ term
        if "PROJ" in zpntest_test.fixterms:
            proj_term_idx = zpntest_test.fixterms.index("PROJ")
            zpntest_test.fixvalues[proj_term_idx] = proj_idx
        else:
            zpntest_test.fixterm(["PROJ"], [proj_idx])

        try:
            if proj_name == "AZP":
                # For AZP, fit only mu (distance parameter), gamma defaults to 0 (no tilt)
                # mu controls projection type: 0=TAN, 1=STG, infinity=SIN
                # Initial: mu=0.1 (close to TAN)

                # Clear all PV2_* terms from both fitterms and fixterms (clean slate for AZP)
                pv2_terms = [f"PV2_{i}" for i in range(1, 10)]
                for term in pv2_terms:
                    # Remove from fitterms
                    if term in zpntest_test.fitterms:
                        idx = zpntest_test.fitterms.index(term)
                        zpntest_test.fitterms.pop(idx)
                        zpntest_test.fitvalues.pop(idx)

                    # Remove from fixterms
                    if term in zpntest_test.fixterms:
                        idx = zpntest_test.fixterms.index(term)
                        zpntest_test.fixterms.pop(idx)
                        zpntest_test.fixvalues.pop(idx)

                # Set PV2_2 (gamma) to 0 and fix it (no tilt)
                zpntest_test.fixterm(["PV2_2"], [0.0])

                # Now add PV2_1 as a fitting term
                zpntest_test.fitterm(["PV2_1"], [0.1])   # mu to be fitted, start close to TAN

                # Do initial fit to check if it's working
                try:
                    refine_fit(zpntest_test, data)
                    refine_fit(zpntest_test, data)
                    final_wssrndf = zpntest_test.wssrndf

                except Exception as e:
                    logging.warning(f"{proj_name}: Fit failed with error: {e}")
                    print(f"{proj_name}: Fit failed with error: {e}")
                    continue

                # Validate the fit
                if not np.isfinite(final_wssrndf) or final_wssrndf <= 0:
                    logging.warning(f"{proj_name}: Invalid WSSR/NDF = {final_wssrndf:.6f}, rejecting")
                    print(f"{proj_name}: Invalid WSSR/NDF = {final_wssrndf:.6f}, rejecting")
                    continue

                # Get the fitted mu value for reporting
                mu_idx = zpntest_test.fitterms.index("PV2_1") if "PV2_1" in zpntest_test.fitterms else None
                mu_value = zpntest_test.fitvalues[mu_idx] if mu_idx is not None else 0.0

                selected_terms = ["PV2_1"]
                msg = f"{proj_name}: WSSR/NDF = {final_wssrndf:.6f} with fitted mu = {mu_value:.6f}"
                logging.info(msg)
                print(msg)

                if final_wssrndf < best_wssrndf:
                    best_wssrndf = final_wssrndf
                    best_projection = proj_name
                    best_zpntest = zpntest_test
                    best_terms = selected_terms
            elif proj_name == "ZPN":
                # For ZPN, use stepwise regression to select polynomial terms
                refine_fit(zpntest_test, data)
                selected_terms = perform_stepwise_astrometry(zpntest_test, data)
                final_wssrndf = zpntest_test.wssrndf

                # If no polynomial terms were selected, this is actually ARC projection
                # ARC (zenithal equidistant): r = u, equivalent to ZPN with only PV2_1=1
                # Widely used for Schmidt telescopes
                if not selected_terms:
                    # Change projection from ZPN to ARC
                    proj_idx_arc = zpntest_test.projections.index("ARC")
                    proj_term_idx = zpntest_test.fixterms.index("PROJ")
                    zpntest_test.fixvalues[proj_term_idx] = proj_idx_arc

                    proj_name = "ARC"
                    msg = f"{proj_name}: WSSR/NDF = {final_wssrndf:.6f} (zenithal equidistant, no distortion)"
                else:
                    msg = f"{proj_name}: WSSR/NDF = {final_wssrndf:.6f} with terms {selected_terms}"

                logging.info(msg)
                print(msg)

                if final_wssrndf < best_wssrndf:
                    best_wssrndf = final_wssrndf
                    best_projection = proj_name
                    best_zpntest = zpntest_test
                    best_terms = selected_terms
            else:
                # For TAN and ZEA, just refine the fit (no free parameters)
                refine_fit(zpntest_test, data)
                refine_fit(zpntest_test, data)
                final_wssrndf = zpntest_test.wssrndf

                msg = f"{proj_name}: WSSR/NDF = {final_wssrndf:.6f}"
                logging.info(msg)
                print(msg)

                if final_wssrndf < best_wssrndf:
                    best_wssrndf = final_wssrndf
                    best_projection = proj_name
                    best_zpntest = zpntest_test
                    best_terms = []

        except Exception as e:
            logging.warning(f"Failed to fit {proj_name} projection: {e}")
            print(f"Failed to fit {proj_name} projection: {e}")
            continue

    # Report the winner
    if best_terms:
        msg = f"Best projection: {best_projection} (WSSR/NDF = {best_wssrndf:.6f}) with terms {best_terms}"
    else:
        msg = f"Best projection: {best_projection} (WSSR/NDF = {best_wssrndf:.6f})"
    logging.info(msg)
    print(msg)

    return best_zpntest

def perform_stepwise_astrometry(zpntest, data, 
                                 # pv_terms=['PV2_3', 'PV2_5', 'PV2_7'],
                                 pv_terms=['PV2_2', 'PV2_3', 'PV2_4', 'PV2_5', 'PV2_6', 'PV2_7'],  # Use if even terms needed
                                 initial_values=None, improvement_threshold=0.001, max_iterations=20):
    """
    Perform bidirectional stepwise regression for ZPN polynomial terms.
    Uses forward selection and backward elimination to find optimal term set.

    Args:
        zpntest: ZPN fitting object (will be modified in place)
        data: PhotometryData with astrometric data
        pv_terms: List of PV2_N terms to consider (default: ['PV2_3', 'PV2_5', 'PV2_7'] - odd terms only)
                  For difficult cases, can use: ['PV2_2', 'PV2_3', 'PV2_4', 'PV2_5', 'PV2_6', 'PV2_7']
        initial_values: Initial guesses for each term (default: odd=Taylor series, even=0)
        improvement_threshold: Minimum relative improvement to keep/remove term (default: 0.001 = 0.1%)
        max_iterations: Maximum forward+backward cycles (default: 20)

    Returns:
        list: Terms that were kept
    """
    if initial_values is None:
        # Default initial values from Taylor series of tan(θ) for odd terms
        # tan(θ) = θ + θ³/3 + 2θ⁵/15 + 17θ⁷/315 + ...
        # Even terms start at 0 (no contribution in ideal TAN projection)
        if len(pv_terms) == 3:
            # Odd terms only (default)
            initial_values = [1.0/3.0, 2.0/15.0, 17.0/315.0]
        else:
            # All terms including even
            initial_values = [0.0, 1.0/3.0, 0.0, 2.0/15.0, 0.0, 17.0/315.0]

    # Ensure we have enough initial values
    while len(initial_values) < len(pv_terms):
        initial_values.append(0.0)

    # Create a dictionary for term -> initial value lookup
    term_init_values = dict(zip(pv_terms, initial_values))

    # Save the initial clean state (before any PV2_* terms are added)
    # This is our baseline to rebuild from
    zpntest_clean = deepcopy(zpntest)

    selected_terms = []
    remaining_terms = list(pv_terms)
    baseline_wssrndf = zpntest.wssrndf

    logging.info(f"Starting bidirectional stepwise astrometry with baseline WSSR/NDF = {baseline_wssrndf:.6f}")
    print(f"Starting bidirectional stepwise astrometry with baseline WSSR/NDF = {baseline_wssrndf:.6f}")

    for iteration in range(max_iterations):
        made_change = False

        # FORWARD STEP: Try adding best remaining term
        best_new_term = None
        best_improvement = 0
        best_new_wssrndf = baseline_wssrndf

        for term in remaining_terms:
            # Try adding this term - start from clean state + selected terms
            zpntest_test = deepcopy(zpntest_clean)
            for t in selected_terms:
                zpntest_test.fitterm([t], [term_init_values[t]])
            zpntest_test.fitterm([term], [term_init_values[term]])

            # Refit with sigma clipping
            try:
                refine_fit(zpntest_test, data)
                refine_fit(zpntest_test, data)
            except Exception as e:
                continue

            new_wssrndf = zpntest_test.wssrndf
            improvement = 1 - new_wssrndf/baseline_wssrndf

            if improvement > best_improvement:
                best_improvement = improvement
                best_new_term = term
                best_new_wssrndf = new_wssrndf

        # Add best term if improvement exceeds threshold
        if best_new_term and best_improvement > improvement_threshold:
            msg = (f"[Iter {iteration+1}] Adding {best_new_term}: WSSR/NDF {baseline_wssrndf:.6f} → "
                   f"{best_new_wssrndf:.6f} (improvement: {best_improvement:.1%})")
            logging.info(msg)
            print(msg)

            selected_terms.append(best_new_term)
            remaining_terms.remove(best_new_term)

            # Rebuild zpntest from clean state with all selected terms
            zpntest_new = deepcopy(zpntest_clean)
            for t in selected_terms:
                zpntest_new.fitterm([t], [term_init_values[t]])
            refine_fit(zpntest_new, data)
            refine_fit(zpntest_new, data)

            # Copy the fitted state back to zpntest
            zpntest.fixterms = zpntest_new.fixterms.copy()
            zpntest.fixvalues = zpntest_new.fixvalues.copy()
            zpntest.fitterms = zpntest_new.fitterms.copy()
            zpntest.fitvalues = zpntest_new.fitvalues.copy()
            zpntest.fiterrors = zpntest_new.fiterrors.copy()
            zpntest.wssrndf = zpntest_new.wssrndf
            zpntest.sigma = zpntest_new.sigma

            baseline_wssrndf = zpntest.wssrndf
            made_change = True

        # BACKWARD STEP: Try removing worst selected term
        if len(selected_terms) > 0:
            worst_term = None
            smallest_degradation = float('inf')
            worst_new_wssrndf = baseline_wssrndf

            for term in selected_terms:
                # Try removing this term - start from clean state
                terms_without = [t for t in selected_terms if t != term]

                # Build from clean state with only the terms we want
                zpntest_test = deepcopy(zpntest_clean)
                for t in terms_without:
                    zpntest_test.fitterm([t], [term_init_values[t]])

                try:
                    refine_fit(zpntest_test, data)
                    refine_fit(zpntest_test, data)
                except Exception as e:
                    continue

                new_wssrndf = zpntest_test.wssrndf
                degradation = 1 - baseline_wssrndf/new_wssrndf

                if degradation < smallest_degradation:
                    smallest_degradation = degradation
                    worst_new_wssrndf = new_wssrndf
                    if degradation < improvement_threshold:
                        worst_term = term

            # Remove term if its removal causes minimal degradation
            if worst_term:
                msg = (f"[Iter {iteration+1}] Removing {worst_term}: WSSR/NDF {baseline_wssrndf:.6f} → "
                       f"{worst_new_wssrndf:.6f} (degradation: {smallest_degradation:.1%})")
                logging.info(msg)
                print(msg)

                selected_terms.remove(worst_term)
                remaining_terms.append(worst_term)

                # Rebuild zpntest from clean state with remaining terms
                zpntest_new = deepcopy(zpntest_clean)
                for t in selected_terms:
                    zpntest_new.fitterm([t], [term_init_values[t]])
                refine_fit(zpntest_new, data)
                refine_fit(zpntest_new, data)

                # Copy the fitted state back to zpntest
                zpntest.fixterms = zpntest_new.fixterms.copy()
                zpntest.fixvalues = zpntest_new.fixvalues.copy()
                zpntest.fitterms = zpntest_new.fitterms.copy()
                zpntest.fitvalues = zpntest_new.fitvalues.copy()
                zpntest.fiterrors = zpntest_new.fiterrors.copy()
                zpntest.wssrndf = zpntest_new.wssrndf
                zpntest.sigma = zpntest_new.sigma

                baseline_wssrndf = zpntest.wssrndf
                made_change = True

        # Stop if no changes in this iteration
        if not made_change:
            logging.info(f"Converged after {iteration+1} iterations")
            print(f"Converged after {iteration+1} iterations")
            break

    if selected_terms:
        logging.info(f"Stepwise astrometry selected terms: {selected_terms}")
        print(f"Stepwise astrometry selected terms: {selected_terms}")
    else:
        logging.info("Stepwise astrometry: no terms improved the fit")
        print("Stepwise astrometry: no terms improved the fit")

    return selected_terms

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
        logging.info(f"CCD_NAME is {camera}")
    except KeyError:
        camera = "C0"
        logging.info(f"CCD_NAME not found, setting {camera}")

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
    elif options.refit_zpn:
        # Refit TAN as ZPN when requested
        logging.info(f"ZPN projection activated via refit_zpn flag (TAN approximation)")
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
    ad = data.get_fitdata('image_x', 'image_y', 'ra', 'dec', 'image_dxy')
    zpntest.fit(ad.astparams)

    # Refine the fit - use stepwise for unknown cameras with refit_zpn
    is_known_camera = camera in ["C1", "C2", "makak", "makak2", "NF4", "ASM1", "ASM-S", "SROT1"]
    is_zpn = (zpntest.fixvalues[zpntest.fixterms.index("PROJ")] == zpntest.projections.index("ZPN"))

    if options.refit_zpn and not is_known_camera and is_zpn:
        # Test TAN, ZEA, AZP, and ZPN projections to find the best one
        logging.info("Testing projection types: TAN, ZEA, AZP (fitted mu), and ZPN with stepwise regression")
        print("Testing projection types: TAN, ZEA, AZP (fitted mu), and ZPN with stepwise regression")

        zpntest = select_best_projection(zpntest, data)

        # Determine the actual projection name
        proj_idx = int(zpntest.fixvalues[zpntest.fixterms.index('PROJ')])
        proj_name = zpntest.projections[proj_idx]

        logging.info(f"Selected projection: {proj_name}")
    else:
        # Normal refinement for known cameras or non-ZPN projections
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

    # Default TAN approximation for unknown cameras when refit_zpn is requested
    # NOTE: Don't set up PV2_3/PV2_5 here - stepwise regression will handle it
    if refit_zpn and camera not in ["C1", "C2", "makak", "makak2", "NF4", "ASM1", "ASM-S", "SROT1"]:
        # Stepwise regression will determine which terms to include
        logging.info(f"ZPN projection will use stepwise regression to determine distortion terms")

#    # ... (similar blocks for other camera types)

