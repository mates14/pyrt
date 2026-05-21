# astrometry_refit.py

import os
import numpy as np
import astropy.wcs
#import astropy.io.fits
from pyrt.core import zpnfit
from pyrt.core.stepwise_regression import ftest_accept
import logging
import matplotlib
matplotlib.use('Agg')
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

def _is_multi_image(zpntest):
    """True if the model has per-image terms for more than one image (any :n with n > 0)."""
    return any(
        ':' in t and t.rsplit(':', 1)[1].isdigit() and int(t.rsplit(':', 1)[1]) > 0
        for t in zpntest.fitterms + zpntest.fixterms
    )

def _get_fitdata_ast(data, zpntest, extra_columns=()):
    """Return (ad, params) for astrometric fitting.

    For multi-image models (any :n term with n>0), includes the 'img' column so
    model() can dispatch per-image CRVAL/CD via fancy indexing.  Single-image
    (bare names or :0 only) uses the 5-tuple backward-compat path.
    """
    base = ('image_x', 'image_y', 'ra', 'dec', 'image_dxy') + tuple(extra_columns)
    if _is_multi_image(zpntest):
        ad = data.get_fitdata(*(base + ('img',)))
        params = ad.astparams_multi
    else:
        ad = data.get_fitdata(*base)
        params = ad.astparams
    return ad, params

def refine_fit(zpntest, data):
    """
    Refine the astrometric fit while maintaining both photometric and astrometric masks.
    Works for both single-image (:0 terms, 5-tuple) and multi-image (:n terms, 6-tuple).
    """
    # Remove all masks to compute residuals for all points
    data.use_mask('default')
    ad_all, params_all = _get_fitdata_ast(data, zpntest)
    # Per-star sigma-clipping: sigma_total = sqrt(sigma_floor² + image_dxy² * variance)
    # sigma_floor (zpntest.sigma) is the systematic WCS floor from the previous fit;
    # image_dxy * sqrt(variance) is the per-object centroiding contribution (same model
    # as compute_object_specific_idlimit in transients.py).
    # For a bad initial fit the floor dominates → permissive threshold for all stars.
    # For a good fit the centroiding term can tighten the threshold for faint stars.
    # Using only dist/err (normalised) collapses to ~0.5 px for bright stars regardless
    # of WCS quality, which discards most of the field and leaves a degenerate corner subset.
    raw_residuals = zpntest.residuals0(zpntest.fitvalues, params_all)
    sigma_floor = zpntest.sigma if np.isfinite(zpntest.sigma) else 0.0
    variance = zpntest.variance if (np.isfinite(zpntest.variance) and zpntest.variance > 1.0) else 1.0
    per_star_sigma = np.sqrt(sigma_floor**2 + ad_all.image_dxy**2 * variance)
    astro_mask = raw_residuals < 3.0 * per_star_sigma

    # Combine the photometric and astrometric masks
    data.use_mask('photometry')
    photo_mask = data.get_current_mask()
    combined_mask = photo_mask & astro_mask

    # Apply the combined mask
    data.add_mask('combined', combined_mask)
    data.use_mask('combined')

    # Refine the fit with the combined mask
    ad_ok, params_ok = _get_fitdata_ast(data, zpntest)
    zpntest.delin = True
    zpntest.fit(params_ok)

def test_crpix_block(zpntest_test, data, proj_name, crpix1_default, crpix2_default):
    """
    F-test (delta_k=2) for whether freeing CRPIX1+CRPIX2 as a block improves the fit.
    If not significant, fixes CRPIX at crpix1_default/crpix2_default (the prior value,
    typically from the image header before any projection testing).
    Returns zpntest_test (possibly with CRPIX now fixed).
    """
    if 'CRPIX1' not in zpntest_test.fitterms or 'CRPIX2' not in zpntest_test.fitterms:
        return zpntest_test  # already fixed, nothing to test

    wssr_free = zpntest_test.wssrndf * zpntest_test.ndf
    ndf_free = zpntest_test.ndf

    zpntest_fixed = deepcopy(zpntest_test)
    zpntest_fixed.fixterm(['CRPIX1', 'CRPIX2'], [crpix1_default, crpix2_default])

    # Blind fit before sigma-clipping: CRPIX shift invalidates current CD/CRVAL,
    # so the tight sigma from the free-CRPIX trial would clip all stars away.
    # Re-establish a global fit on all photometry-matched stars first.
    data.use_mask('photometry')
    _, params = _get_fitdata_ast(data, zpntest_fixed)
    zpntest_fixed.delin = False
    zpntest_fixed.fit(params)

    refine_fit(zpntest_fixed, data)
    refine_fit(zpntest_fixed, data)

    wssr_fixed = zpntest_fixed.wssrndf * zpntest_fixed.ndf
    ndf_fixed = zpntest_fixed.ndf

    if not np.isfinite(wssr_fixed) or ndf_fixed <= 0:
        # Fixed-CRPIX fit degenerate; keep CRPIX free
        msg = f"{proj_name}: Fixed-CRPIX fit degenerate (NDF={ndf_fixed}), keeping CRPIX free"
        logging.warning(msg)
        print(msg)
        return zpntest_test

    if ftest_accept(wssr_fixed, wssr_free, ndf_free, delta_k=2):
        crpix1 = zpntest_test.termval('CRPIX1')
        crpix2 = zpntest_test.termval('CRPIX2')
        msg = f"{proj_name}: CRPIX free ({crpix1:.1f}, {crpix2:.1f}) is significant (F-test δk=2)"
        logging.info(msg)
        print(msg)
        return zpntest_test
    else:
        msg = (f"{proj_name}: CRPIX not significant (F-test δk=2), "
               f"fixed at prior ({crpix1_default:.1f}, {crpix2_default:.1f})")
        logging.info(msg)
        print(msg)
        return zpntest_fixed

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

    # Save CRPIX prior before any projection fitting; used as the fixed value when
    # the F-test says CRPIX does not improve the fit (avoids fixing at a noisy fitted position)
    crpix1_prior = zpntest.termval('CRPIX1')
    crpix2_prior = zpntest.termval('CRPIX2')

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
                    zpntest_test = test_crpix_block(zpntest_test, data, proj_name, crpix1_prior, crpix2_prior)
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

                # Validate that astropy can load this WCS
                # Use relax=True to allow valid non-standard WCS configurations
                try:
                    test_wcs = astropy.wcs.WCS(zpntest_test.wcs(), relax=True)
                    wcs_valid = True
                except (ValueError, RuntimeError) as e:
                    wcs_valid = False
                    logging.warning(f"{proj_name}: WCS validation failed - {e}")
                    print(f"{proj_name}: WCS validation failed, rejecting")

                if wcs_valid and final_wssrndf < best_wssrndf:
                    best_wssrndf = final_wssrndf
                    best_projection = proj_name
                    best_zpntest = zpntest_test
                    best_terms = selected_terms
            elif proj_name == "ZPN":
                # For ZPN, use stepwise regression to select polynomial terms.
                # Two refine_fit passes before stepwise normalize the baseline sigma
                # to the same level as candidate models (which also do two passes each),
                # so the forward F-test comparison is apples-to-apples.
                refine_fit(zpntest_test, data)
                refine_fit(zpntest_test, data)
                selected_terms = perform_stepwise_astrometry(zpntest_test, data)
                zpntest_test = test_crpix_block(zpntest_test, data, proj_name, crpix1_prior, crpix2_prior)
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

                # Validate that astropy can load this WCS
                # This is crucial for ZPN as invalid PV parameters can cause astropy to fail
                # Use relax=True to allow valid non-standard WCS configurations (esp. ZPN+SIP)
                try:
                    test_wcs = astropy.wcs.WCS(zpntest_test.wcs(), relax=True)
                    # Also test that we can actually do transformations
                    test_wcs.all_pix2world(100, 100, 0)
                    wcs_valid = True
                except (ValueError, RuntimeError, Exception) as e:
                    wcs_valid = False
                    logging.warning(f"{proj_name}: WCS validation failed - {e}")
                    print(f"{proj_name}: WCS validation failed, rejecting")

                if wcs_valid and final_wssrndf < best_wssrndf:
                    best_wssrndf = final_wssrndf
                    best_projection = proj_name
                    best_zpntest = zpntest_test
                    best_terms = selected_terms
            else:
                # For TAN and ZEA, just refine the fit (no free parameters)
                refine_fit(zpntest_test, data)
                refine_fit(zpntest_test, data)
                zpntest_test = test_crpix_block(zpntest_test, data, proj_name, crpix1_prior, crpix2_prior)
                final_wssrndf = zpntest_test.wssrndf

                msg = f"{proj_name}: WSSR/NDF = {final_wssrndf:.6f}"
                logging.info(msg)
                print(msg)

                # Validate that astropy can load this WCS
                # Use relax=True to allow valid non-standard WCS configurations
                try:
                    test_wcs = astropy.wcs.WCS(zpntest_test.wcs(), relax=True)
                    wcs_valid = True
                except (ValueError, RuntimeError) as e:
                    wcs_valid = False
                    logging.warning(f"{proj_name}: WCS validation failed - {e}")
                    print(f"{proj_name}: WCS validation failed, rejecting")

                if not np.isfinite(final_wssrndf) or final_wssrndf <= 0:
                    logging.warning(f"{proj_name}: Invalid WSSR/NDF = {final_wssrndf:.6f}, rejecting")
                    print(f"{proj_name}: Invalid WSSR/NDF = {final_wssrndf:.6f}, rejecting")
                elif wcs_valid and final_wssrndf < best_wssrndf:
                    best_wssrndf = final_wssrndf
                    best_projection = proj_name
                    best_zpntest = zpntest_test
                    best_terms = []

        except Exception as e:
            logging.warning(f"Failed to fit {proj_name} projection: {e}")
            print(f"Failed to fit {proj_name} projection: {e}")
            continue

    # Report the winner
    if best_zpntest is None:
        msg = "ERROR: All projection types failed validation. Cannot create valid WCS."
        logging.error(msg)
        print(msg)
        return None

    if best_terms:
        msg = f"Best projection: {best_projection} (WSSR/NDF = {best_wssrndf:.6f}) with terms {best_terms}"
    else:
        msg = f"Best projection: {best_projection} (WSSR/NDF = {best_wssrndf:.6f})"
    logging.info(msg)
    print(msg)

    return best_zpntest

def perform_stepwise_astrometry(zpntest, data,
                                 # pv_terms=['PV2_3', 'PV2_5', 'PV2_7'],
                                 pv_terms=['PV2_2', 'PV2_3', 'PV2_4', 'PV2_5', 'PV2_6', 'PV2_7'],
                                 initial_values=None, max_iterations=20):
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
    baseline_wssr = zpntest.wssrndf * zpntest.ndf
    baseline_ndf = zpntest.ndf

    logging.info(f"Starting bidirectional stepwise astrometry with baseline WSSR/NDF = {baseline_wssrndf:.6f}")
    print(f"Starting bidirectional stepwise astrometry with baseline WSSR/NDF = {baseline_wssrndf:.6f}")

    for iteration in range(max_iterations):
        made_change = False

        # FORWARD STEP: Try adding best remaining term
        best_new_term = None
        best_F = 0.0
        best_new_wssrndf = baseline_wssrndf
        best_new_wssr = baseline_wssr
        best_new_ndf = baseline_ndf

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

            new_wssr = zpntest_test.wssrndf * zpntest_test.ndf
            new_ndf = zpntest_test.ndf
            if ftest_accept(baseline_wssr, new_wssr, new_ndf):
                F_stat = (baseline_wssr - new_wssr) / (new_wssr / new_ndf)
                if F_stat > best_F:
                    best_F = F_stat
                    best_new_term = term
                    best_new_wssrndf = zpntest_test.wssrndf
                    best_new_wssr = new_wssr
                    best_new_ndf = new_ndf

        # Add best term if F-test passed
        if best_new_term:
            msg = (f"[Iter {iteration+1}] Adding {best_new_term}: WSSR/NDF {baseline_wssrndf:.6f} → "
                   f"{best_new_wssrndf:.6f} (F={best_F:.1f}, NDF={best_new_ndf})")
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
            zpntest.ndf = zpntest_new.ndf
            zpntest.sigma = zpntest_new.sigma

            baseline_wssrndf = zpntest.wssrndf
            baseline_wssr = zpntest.wssrndf * zpntest.ndf
            baseline_ndf = zpntest.ndf
            made_change = True

        # BACKWARD STEP: Try removing worst selected term
        if len(selected_terms) > 0:
            worst_term = None
            smallest_F = float('inf')
            worst_new_wssrndf = baseline_wssrndf
            worst_new_wssr = baseline_wssr
            worst_new_ndf = baseline_ndf

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

                new_wssr = zpntest_test.wssrndf * zpntest_test.ndf
                new_ndf = zpntest_test.ndf
                F_stat = (new_wssr - baseline_wssr) / (baseline_wssr / baseline_ndf) \
                    if (new_wssr > baseline_wssr and baseline_ndf > 0) else 0.0
                if F_stat < smallest_F:
                    smallest_F = F_stat
                    worst_new_wssrndf = zpntest_test.wssrndf
                    worst_new_wssr = new_wssr
                    worst_new_ndf = new_ndf
                    # Only remove if wssr actually increased (guard against sigma-clipping artifacts)
                    if new_wssr > baseline_wssr and not ftest_accept(new_wssr, baseline_wssr, baseline_ndf):
                        worst_term = term

            # Remove term if F-test says it is not significant
            if worst_term:
                msg = (f"[Iter {iteration+1}] Removing {worst_term}: WSSR/NDF {baseline_wssrndf:.6f} → "
                       f"{worst_new_wssrndf:.6f} (F={smallest_F:.2f}, not significant at α=0.05)")
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
                zpntest.ndf = zpntest_new.ndf
                zpntest.sigma = zpntest_new.sigma

                baseline_wssrndf = zpntest.wssrndf
                baseline_wssr = zpntest.wssrndf * zpntest.ndf
                baseline_ndf = zpntest.ndf
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

def compute_error_model(zpntest, data):
    """
    Fit sigma_total² = S0 + SC·(ERRX2+ERRY2) from astrometric residuals.
    centering2 = image_var = ERRX2_IMAGE + ERRY2_IMAGE (total centroid variance, pix²).
    Returns (sqrt(S0), SC) suitable for ASTSIGMA and ASTVAR headers.
    Uses the default mask (all matched stars, not just sigma-clipped).
    Slope estimated via Theil-Sen (median of pairwise slopes) for outlier robustness.
    """
    data.use_mask('default')
    ad, params = _get_fitdata_ast(data, zpntest, extra_columns=('image_var',))
    residuals_px = zpntest.residuals0(zpntest.fitvalues, params)
    centering2   = ad.image_var   # ERRX2 + ERRY2 in pix²

    order = np.argsort(centering2)
    r2 = residuals_px[order] ** 2
    c2 = centering2[order]
    n  = len(r2)

    if n < 10:
        S0 = float(np.median(r2))
        return np.sqrt(max(S0, 0.0)), 1.0

    nbins = min(10, n // 5)
    bins  = np.array_split(np.arange(n), nbins)
    med_r2 = np.array([np.median(r2[b]) for b in bins])
    med_c2 = np.array([np.median(c2[b]) for b in bins])

    # Theil-Sen slope: median of all pairwise slopes — robust to outlier bins
    slopes = []
    for i in range(len(med_c2)):
        for j in range(i + 1, len(med_c2)):
            dc = med_c2[j] - med_c2[i]
            if abs(dc) > 1e-12:
                slopes.append((med_r2[j] - med_r2[i]) / dc)
    if slopes:
        SC = max(float(np.median(slopes)), 0.0)
        S0 = max(float(np.median(med_r2 - SC * med_c2)), 0.0)
    else:
        S0 = max(float(np.median(med_r2)), 0.0)
        SC = 0.0

    # Actual scatter of all matched stars (telescope-quality metric for DB upload decisions).
    # Uses the default mask so it reflects the full matched set, not just sigma-clipped stars.
    scatter = float(np.median(np.abs(residuals_px)) / 0.67)

    logging.info(f"Error model fit: S0={S0:.6f} px², SC={SC:.4f}, ASTSIGMA={np.sqrt(S0):.4f} px, ASTSCATT={scatter:.4f} px")
    return np.sqrt(S0), SC, scatter


def refit_astrometry_multi(alldet, data, options):
    """Fit astrometry simultaneously for multiple images.

    Per-image terms (CRVAL1:n, CRVAL2:n, CDi_j:n) describe pointing and rotation
    for image n.  Global terms (CRPIX1, CRPIX2, PV2_*) are shared across images
    and represent the fixed optical model of the camera.

    Fitting many frames together strongly constrains the optical axis (CRPIX) and
    radial distortion (PV2_*) because frame-to-frame pointing dither breaks the
    degeneracy between CRVAL and PV2_1 that plagues single-image fits.

    Parameters
    ----------
    alldet : list of astropy.Table
        One detection table per image (each with a FITS header in .meta).
    data : PhotometryData
        Combined star catalogue from all images; img column holds 0-based image index.
    options : argparse.Namespace
        Same options as refit_astrometry (refit_zpn, szp, save_wcs, …).

    Returns
    -------
    zpnfit.zpnfit
        Fitted model with per-image CRVAL/CD terms and global CRPIX/PV2_* terms.
    """
    det0 = alldet[0]
    camera = det0.meta.get('CCD_NAME', 'C0')
    telescope = str(det0.meta.get('TELESCOP', ''))
    n_images = len(alldet)

    msg = f"Multi-image astrometry: {n_images} images, camera={camera}"
    logging.info(msg); print(msg)

    # --- Build the shared model (projection + global optical terms) ---
    if options.szp:
        zpntest = zpnfit.zpnfit(proj="AZP")
        zpntest.fitterm(["PV2_1"], [1])
    elif options.refit_zpn:
        if camera in ["C0", "C1", "C2", "makak", "makak2", "NF4", "ASM1", "ASM-S", "SROT1"]:
            zpntest = zpnfit.zpnfit(proj="ZPN")
            zpntest.fixterm(["PV2_1"], [1])
        elif camera in ["CAM-ZEA"]:
            zpntest = zpnfit.zpnfit(proj="ZEA")
        else:
            zpntest = zpnfit.zpnfit(proj="ZPN")
            zpntest.fixterm(["PV2_1"], [1])
        setup_camera_params(zpntest, camera, options.refit_zpn, telescope, meta=det0.meta)
    else:
        # Gentle refit: projection and global terms from first image's header
        ctype1 = str(det0.meta.get('CTYPE1', ''))
        if 'ZPN' in ctype1:
            zpntest = zpnfit.zpnfit(proj="ZPN")
            zpntest.fixterm(["PV2_1"], [1])
        elif 'ZEA' in ctype1:
            zpntest = zpnfit.zpnfit(proj="ZEA")
        else:
            zpntest = zpnfit.zpnfit(proj="TAN")
        # Global: fix CRPIX at first image's header value
        for term in ['CRPIX1', 'CRPIX2']:
            if term in det0.meta:
                zpntest.fixterm([term], [float(det0.meta[term])])
        # Global: fix any existing distortion coefficients at first image's header values
        for key in sorted(det0.meta.keys()):
            if isinstance(key, str) and key.startswith('PV2_') and key != 'PV2_1':
                try:
                    zpntest.fixterm([key], [float(det0.meta[key])])
                except (TypeError, ValueError):
                    pass

    # --- Per-image terms: CRVAL and CD matrix for each frame ---
    for n, det in enumerate(alldet):
        keys_invalid = False
        for base_term, key in [("CRVAL1", "CRVAL1"), ("CRVAL2", "CRVAL2"),
                                ("CD1_1", "CD1_1"), ("CD1_2", "CD1_2"),
                                ("CD2_1", "CD2_1"), ("CD2_2", "CD2_2")]:
            term = f"{base_term}:{n}"
            try:
                zpntest.fitterm([term], [det.meta[key]])
            except KeyError:
                keys_invalid = True
        if keys_invalid:
            logging.warning(f"Image {n} ({det.meta.get('FITSFILE', '?')}): some WCS keys missing")
        # For refit_zpn, also fit CRPIX per-session if not already set globally
        # (camera-specific CRPIX already handled by setup_camera_params above)

    if options.refit_zpn and 'CRPIX1' not in zpntest.fitterms and 'CRPIX1' not in zpntest.fixterms:
        # Camera not known: fit global CRPIX from first image header
        for term in ['CRPIX1', 'CRPIX2']:
            if term in det0.meta:
                zpntest.fitterm([term], [float(det0.meta[term])])

    # --- Initial global fit on all photometry-matched stars ---
    data.use_mask('photometry')
    ad = data.get_fitdata('image_x', 'image_y', 'ra', 'dec', 'image_dxy', 'img')
    zpntest.delin = False
    zpntest.fit(ad.astparams_multi)

    # --- Refine with sigma-clipping (auto-detects multi-image via _is_multi_image) ---
    refine_fit(zpntest, data)
    refine_fit(zpntest, data)

    print(zpntest)

    # --- Error model ---
    zpntest.sigma, zpntest.variance, zpntest.scatter = compute_error_model(zpntest, data)
    print(f"Error model: ASTSIGMA={zpntest.sigma:.4f} px (floor), "
          f"ASTVAR={zpntest.variance:.4f}, ASTSCATT={zpntest.scatter:.4f} px")

    return zpntest


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

    telescope = str(det.meta.get('TELESCOP', ''))

    if options.szp:
        zpntest = zpnfit.zpnfit(proj="AZP")
        zpntest.fitterm(["PV2_1"], [1])

    elif options.refit_zpn:
        # Full refit from camera calibration priors (-z flag).
        # Ignores header CTYPE/CRPIX and uses hardcoded camera model.
        if camera in ["C0", "C1", "C2", "makak", "makak2", "NF4", "ASM1", "ASM-S", "SROT1"]:
            logging.info(f"ZPN projection activated (refit_zpn)")
            zpntest = zpnfit.zpnfit(proj="ZPN")
            zpntest.fixterm(["PV2_1"], [1])
        elif camera in ["CAM-ZEA"]:
            zpntest = zpnfit.zpnfit(proj="ZEA")
        else:
            logging.info(f"ZPN projection activated via refit_zpn flag (unknown camera)")
            zpntest = zpnfit.zpnfit(proj="ZPN")
            zpntest.fixterm(["PV2_1"], [1])

    else:
        # Gentle refit (-a without -z): respect the existing WCS structure.
        # Read projection from CTYPE1, fix CRPIX and distortion terms at header values.
        # Only CD matrix and CRVAL are free.  Works correctly for both native ZPN images
        # and reprojected TAN images regardless of CCD_NAME.
        ctype1 = str(det.meta.get('CTYPE1', ''))
        if 'ZPN' in ctype1:
            logging.info(f"Gentle refit: ZPN projection from header CTYPE1={ctype1!r}")
            zpntest = zpnfit.zpnfit(proj="ZPN")
            zpntest.fixterm(["PV2_1"], [1])
        elif 'ZEA' in ctype1:
            logging.info(f"Gentle refit: ZEA projection from header CTYPE1={ctype1!r}")
            zpntest = zpnfit.zpnfit(proj="ZEA")
        else:
            logging.info(f"Gentle refit: TAN projection from header CTYPE1={ctype1!r}")
            zpntest = zpnfit.zpnfit(proj="TAN")

    # Set up initial WCS parameters (CD, CRVAL, CRPIX) from header as fit terms
    keys_invalid = setup_initial_wcs(zpntest, det.meta)

    if keys_invalid:
        logging.warning("I do not understand the WCS to be fitted, skipping...")
        return None

    if options.refit_zpn or options.szp:
        # Full refit: apply hardcoded camera-specific CRPIX and distortion priors
        setup_camera_params(zpntest, camera, options.refit_zpn, telescope, meta=det.meta)
    else:
        # Gentle refit: fix CRPIX at header values (already loaded by setup_initial_wcs)
        for term in ['CRPIX1', 'CRPIX2']:
            if term in det.meta:
                zpntest.fixterm([term], [float(det.meta[term])])
        # Fix any existing ZPN distortion terms (PV2_3, PV2_5, …) at their header values
        for key in sorted(det.meta.keys()):
            if isinstance(key, str) and key.startswith('PV2_') and key != 'PV2_1':
                try:
                    zpntest.fixterm([key], [float(det.meta[key])])
                except (TypeError, ValueError):
                    pass

    data.use_mask('photometry')

    # Perform the initial fit
    _, params = _get_fitdata_ast(data, zpntest)
    zpntest.fit(params)

    # Refine the fit - use stepwise for unknown cameras with refit_zpn
    is_known_camera = (camera in ["C1", "C2", "makak", "makak2", "NF4", "ASM1", "ASM-S", "SROT1"] or
                       (camera == "C0" and telescope == "D50"))
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
        # Save state before adding SIP in case we need to fall back
        zpntest_before_sip = deepcopy(zpntest)

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

        # Validate the final Projection+SIP combination
        # Some specific combinations of PV and SIP coefficients can fail WCSLIB validation
        # even though they're mathematically valid
        try:
            test_wcs = astropy.wcs.WCS(zpntest.wcs(), relax=True)
            test_wcs.all_pix2world(100, 100, 0)  # Test transformation works
            logging.info("Final WCS validation passed (projection+SIP)")
        except (ValueError, RuntimeError, Exception) as e:
            logging.warning(f"Final WCS validation failed after adding SIP: {e}")
            logging.warning("Falling back to projection without SIP distortion")
            print(f"WARNING: Projection+SIP combination failed validation, using projection without SIP")
            # Restore state before SIP was added
            zpntest = zpntest_before_sip

    # Save WCS solution if requested
    if options.save_wcs:
        if options.save_wcs is True:
            # Use input filename with .wcs extension
            base_filename = os.path.splitext(det.meta['FITSFILE'])[0]
            wcs_filename = f"{base_filename}.wcs"
        else:
            wcs_filename = options.save_wcs
        zpntest.write_wcs(wcs_filename)

    print(zpntest)

    # Astrometric arrow plot now generated in dophot.py using plot_astrometric_arrows()

    # Fit the S0+SC error model and overwrite sigma/variance so that
    # zpnfit.write() stores correct ASTSIGMA=sqrt(S0) and ASTVAR=SC semantics.
    # scatter = actual median scatter — the old ASTSIGMA semantics, used for DB upload quality check.
    zpntest.sigma, zpntest.variance, zpntest.scatter = compute_error_model(zpntest, data)
    print(f"Error model: ASTSIGMA={zpntest.sigma:.4f} px (floor), ASTVAR={zpntest.variance:.4f}, ASTSCATT={zpntest.scatter:.4f} px (scatter)")

    return zpntest

def setup_initial_wcs(zpntest, meta):
    """Set up initial WCS parameters.

    Per-image terms (CRVAL, CD matrix) use the :0 suffix so the model can handle
    multi-frame fitting where each image has its own pointing and rotation but all
    images share the same optical axis (CRPIX) and distortion (PV2_*).
    """
    keys_invalid = False
    # Per-image terms with :0 suffix (CRVAL and CD matrix vary per frame)
    for term, key in [("CRVAL1:0", "CRVAL1"), ("CRVAL2:0", "CRVAL2"),
                      ("CD1_1:0", "CD1_1"), ("CD1_2:0", "CD1_2"),
                      ("CD2_1:0", "CD2_1"), ("CD2_2:0", "CD2_2")]:
        try:
            zpntest.fitterm([term], [meta[key]])
        except KeyError:
            keys_invalid = True
    # Global terms (shared across all images)
    for term in ["CRPIX1", "CRPIX2"]:
        try:
            zpntest.fitterm([term], [meta[term]])
        except KeyError:
            keys_invalid = True

    if keys_invalid:
        try: # CROTA1 & 2 are optional, may be left to default=0 in stacked images
            crota1 = meta['CROTA1'] * np.pi / 180
        except:
            crota1 = 0
        try:
            crota2 = meta['CROTA2'] * np.pi / 180
        except:
            crota2 = 0
        try:
            # Try to interpret old-fashioned WCS with CROTA
            zpntest.fitterm(['CD1_1:0'], [meta['CDELT1'] * np.cos(crota1)])
            zpntest.fitterm(['CD1_2:0'], [meta['CDELT1'] * np.sin(crota1)])
            zpntest.fitterm(['CD2_1:0'], [meta['CDELT2'] * -np.sin(crota2)])
            zpntest.fitterm(['CD2_2:0'], [meta['CDELT2'] * np.cos(crota2)])
            keys_invalid = False
        except KeyError:
            keys_invalid = True

    return keys_invalid

def _crpix_for_crop(crpix1, crpix2, meta):
    """Adjust full-frame CRPIX values to sub-frame pixel coordinates using LTV/LTM."""
    ltv1  = meta.get('LTV1',  0.0)
    ltv2  = meta.get('LTV2',  0.0)
    ltm11 = meta.get('LTM1_1', 1.0)
    ltm22 = meta.get('LTM2_2', 1.0)
    if ltm11 == 0:
        ltm11 = 1.0
    if ltm22 == 0:
        ltm22 = 1.0
    return (crpix1 - ltv1) / ltm11, (crpix2 - ltv2) / ltm22


def setup_camera_params(zpntest, camera, refit_zpn, telescope='', meta=None):
    """Set up camera-specific parameters.

    meta is the FITS header dict; when present, CRPIX values are corrected
    for sub-frame crops using the LTV1/LTV2/LTM1_1/LTM2_2 keywords.
    """
    if meta is None:
        meta = {}

    def crpix(cx, cy):
        return _crpix_for_crop(cx, cy, meta)

    if camera == "C0" and telescope == "D50":
        if refit_zpn:
            zpntest.fitterm(["PV2_3"], [300])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(543, 530)))
        else:
            zpntest.fixterm(["PV2_3"], [300])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(543, 530)))

    if camera == "C1":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [7.5, 386.1])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(2090, 2043)))
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [7.5, 386.1])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(2090, 2043)))

    if camera == "C2":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [8.255, 343.8])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(2124.0, 2039.0)))
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [8.255, 343.8])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(2124.0, 2039.0)))

    if camera == "SROT1":
        if refit_zpn:
            logging.info(f"SROT1 setup being loaded (active)")
            zpntest.fitterm(["PV2_3", "PV2_5"], [38.561185, 3461.163423])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(1882.796706, 2055.012734)))
        else:
            logging.info(f"SROT1 setup being loaded (passive)")
            zpntest.fixterm(["PV2_3", "PV2_5"], [38.561185, 3461.163423])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(1882.796706, 2055.012734)))

    if camera in ( "makak2",  "makak"):
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [0.132, 0.569])
#            zpntest.fitterm(["PV2_3", "PV2_5"], [0.131823, 0.282538])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(813.6, 622.8)))
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [0.132, 0.569])
#            zpntest.fixterm(["PV2_3", "PV2_5"], [0.131823, 0.282538])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(813.6, 622.8)))

    if camera == "NF4":
        zpntest.fitterm(["PV2_3"], [65.913305900171])
        zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(522.75, 569.96)))

    if camera == "ASM1":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5", "PV2_7"], [-0.0388566,0.001255,-0.002769])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(2054.5, 2059.0)))
        else:
            zpntest.fixterm(["PV2_3", "PV2_5", "PV2_7"], [-0.0388566,0.001255,-0.002769])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(2054.5, 2059.0)))
    if camera == "ASM-S":
        if refit_zpn:
            zpntest.fitterm(["PV2_3", "PV2_5"], [-0.0456,0.0442])
            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(128.5, 128.5)))
#            zpntest.fitterm(["CRPIX1", "CRPIX2"], list(crpix(339.8, 335.5)))
        else:
            zpntest.fixterm(["PV2_3", "PV2_5"], [-0.0456,0.0442])
            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(128.5, 128.5)))
#            zpntest.fixterm(["CRPIX1", "CRPIX2"], list(crpix(339.8, 335.5)))
# PV2_3   =  -0.045557711245 / ± 0.001318165008 (2.893396%)
# PV2_5   =   0.044189874766 / ± 0.002482740922 (5.618348%)
# CRPIX1  = 339.882095473812 / ± 0.262750191549 (0.077306%)
# CRPIX2  = 335.523756136020 / ± 0.325225858217 (0.096931%)

    # For unknown cameras (including C0 on non-D50 telescopes): stepwise regression selects terms.
    known = ["C1", "C2", "makak", "makak2", "NF4", "ASM1", "ASM-S", "SROT1"]
    if refit_zpn and not (camera in known or (camera == "C0" and telescope == "D50")):
        # Stepwise regression will determine which terms to include
        logging.info(f"ZPN projection will use stepwise regression to determine distortion terms")

#    # ... (similar blocks for other camera types)

