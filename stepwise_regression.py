#!/usr/bin/env python3
"""
Stepwise regression module for photometric model fitting.
Includes functions for term expansion, model setup, and bidirectional stepwise regression.
"""

import numpy as np
import concurrent.futures
from copy import deepcopy

def expand_pseudo_term(term):
    """
    Expands pseudo-terms like '.p3' or '.r2' into their constituent terms.

    Args:
        term (str): The term to expand (e.g., '.p3' or '.r2')

    Returns:
        list: List of expanded terms
    """
    expanded_terms = []

    if term[0] == '.':
        if term[1] == 'p':
            # Surface polynomial
            pol_order = int(term[2:])
            for pp in range(1, pol_order + 1):
                for rr in range(0, pp + 1):
                    expanded_terms.append(f"P{rr}X{pp-rr}Y")
        if term[1] == 's':
            # subpixel variation
            expanded_terms.append("SXY")
            expanded_terms.append("SX")
            expanded_terms.append("SY")
        if term[1] == 'l':
            # nonlinearity with RC/RS/RO (r is radius, needed something else)
            expanded_terms.append("RC")
            expanded_terms.append("RS")
            expanded_terms.append("RO")
        elif term[1] in ['r', 'c', 'd', 'e', 'f', 'n']:
            # Radial polynomial
            pol_order = int(term[2:])
            for pp in range(1, pol_order + 1):
                expanded_terms.append(f"P{pp}{term[1].upper()}")
    else:
        # Regular term, no expansion needed
        expanded_terms.append(term)

    return expanded_terms

def parse_terms(terms_string):
    """
    Parse the terms string into a list of individual terms.

    Args:
        terms_string (str): Comma-separated string of terms.

    Returns:
        list: List of individual terms.
    """
    return [term.strip() for term in terms_string.split(',') if term.strip()]

def setup_terms(ffit, terms, initial_values=None):
    """
    Set up terms in the fitting object with optional initial values.

    Args:
        ffit: The fitting object
        terms (list): List of terms to set up
        initial_values (dict): Dictionary of initial values for terms
    """
    ffit.fixall()  # Reset all terms

    if not terms:
        return

    # Use initial values from model when available
    if initial_values:
        values = [initial_values.get(term, 1e-6) for term in terms]
    else:
        values = [1e-6] * len(terms)

    # Add all terms as fitting terms
    ffit.fitterm(terms, values=values)

def try_term_robust(ffit, term, current_terms, fdata, initial_values=None):
    """Try fitting with a new term or subset of terms, using robust statistics"""
    new_ffit = deepcopy(ffit)

    # Determine which terms to use
    terms_to_fit = current_terms + [term] if term is not None else current_terms

    # Create dictionary of initial values
    term_values = {t: 1e-6 for t in terms_to_fit}
    if initial_values:
        term_values.update({k: v for k, v in initial_values.items() if k in terms_to_fit})

    setup_terms(new_ffit, terms_to_fit, term_values)

    # Make a copy of the input data
    thread_data = tuple(np.copy(arr) for arr in fdata)

    # Iterative fitting with outlier removal
    max_iterations = 3
    current_mask = np.ones(len(thread_data[0]), dtype=bool)
    prev_wssrndf = float('inf')

    for iteration in range(max_iterations):
        # Apply current mask to data
        masked_data = tuple(arr[current_mask] for arr in thread_data)

        # Perform the fit
        try:
            new_ffit.fit(masked_data)
        except Exception as e:
            print(f"Fitting failed for terms {terms_to_fit}: {str(e)}")
            return float('inf'), None

        # Calculate residuals on all points
        residuals = new_ffit.residuals(new_ffit.fitvalues, thread_data)

        # Create new mask for points within 5-sigma
        new_mask = np.abs(residuals) < 5 * new_ffit.wssrndf

        # Check for convergence
        if np.array_equal(new_mask, current_mask) or new_ffit.wssrndf >= prev_wssrndf:
            break

        current_mask = new_mask
        prev_wssrndf = new_ffit.wssrndf

    return new_ffit.wssrndf, current_mask

def try_remove_term_parallel(term_to_remove, current_terms, ffit, fdata, initial_values):
    """Try removing a term in parallel context"""
    test_terms = [t for t in current_terms if t != term_to_remove]
    new_wssrndf, _ = try_term_robust(ffit, None, test_terms, fdata, initial_values)
    return term_to_remove, new_wssrndf

def perform_stepwise_regression(data, ffit, initial_terms, options, metadata):
    """
    Perform bidirectional stepwise regression while maintaining robustness to outliers.
    Uses parallel processing for both forward and backward steps.

    Args:
        data: PhotometryData object containing the data
        ffit: fotfit object for model fitting
        initial_terms: List of initial terms to consider
        options: Command line options
        metadata: List of metadata for the images

    Returns:
        tuple: (selected_terms, final_wssrndf)
    """
    # Initialize with just zeropoints
    selected_terms = []
    remaining_terms = initial_terms.copy()

    # Get initial fit data
    fdata = data.get_arrays('y', 'adif', 'coord_x', 'coord_y', 'color1',
                           'color2', 'color3', 'color4', 'img', 'x', 'dy',
                           'image_x', 'image_y')

    # Initial fit
    ffit.fixall()
    ffit.fit(fdata) # i.e. this only fits the zeropoint and optionally PX,PY if fit_xy
    best_wssrndf = ffit.wssrndf

    # Load initial values from model if available
    initial_values = {}
    if options.model:
        for term, value in zip(ffit.fixterms + ffit.fitterms,
                             ffit.fixvalues + ffit.fitvalues):
            initial_values[term] = value

    max_iterations = 100  # Prevent infinite loops
    iteration = 0
    improvement_threshold = 0.001  # Minimum relative improvement to accept a term

    while iteration < max_iterations:
#        print(f"At beginning: wssrndf={best_wssrndf}")
        iteration += 1
        made_change = False

        # Forward step - try adding terms in parallel
        best_new_term = None
        best_improvement = 0
        best_new_mask = None

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all term trials in parallel
            future_to_term = {
                executor.submit(try_term_robust, ffit, term, selected_terms, fdata, initial_values): term
                for term in remaining_terms
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_term):
                term = future_to_term[future]
                try:
                    new_wssrndf, new_mask = future.result()
                    improvement = 1 - new_wssrndf/best_wssrndf

                    if improvement > improvement_threshold and improvement > best_improvement:
                        best_improvement = improvement
                        best_new_term = term
                        best_new_mask = new_mask
                        best_new_wssrndf = new_wssrndf
                except Exception as e:
                    print(f"Error trying term {term}: {str(e)}")

#        print(f"After search for new: wssrndf={best_wssrndf}")
        # Add best term if found
        if best_new_term:
            selected_terms.append(best_new_term)
            remaining_terms.remove(best_new_term)
            best_wssrndf = best_new_wssrndf  # Simply use the new value
            made_change = True
            print(f"Added term {best_new_term} (improvement: {best_improvement:.1%}). New wssrndf: {best_wssrndf}")

            # Apply the new mask to data
            data.apply_mask(best_new_mask, 'current')

#            print(f"Before removal step: wssrndf={best_wssrndf}")
            # Backward step - check if any terms can be removed in parallel
            if len(selected_terms) > 1:  # Keep at least one term
                worst_term = None
                smallest_degradation = float('inf')

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    # Submit all term removal trials in parallel
                    futures = [
                        executor.submit(try_remove_term_parallel,
                                     term, selected_terms,
                                     ffit, fdata, initial_values)
                        for term in selected_terms
                    ]

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            term, new_wssrndf = future.result()
                            degradation = 1 - best_wssrndf/new_wssrndf

                            if degradation < improvement_threshold and degradation < smallest_degradation:
                                smallest_degradation = degradation
                                worst_term = term
                                worst_new_wssrndf = new_wssrndf
                        except Exception as e:
                            print(f"Error trying to remove term: {str(e)}")

#                print(f"Before actually removing: wssrndf={best_wssrndf}")
                # Remove term if its contribution is minimal
                if worst_term:
                    print(f"Removing term {worst_term} (degradation: {smallest_degradation:.1%})")
                    ffit.removeterm(worst_term)
#                    ffit.fit(fdata)
                    best_wssrndf = worst_new_wssrndf
#                    best_wssrndf = ffit.wssrndf
                    selected_terms.remove(worst_term)
                    remaining_terms.append(worst_term)
                    made_change = True
                    #selected_terms.remove(worst_term)
                    #remaining_terms.append(worst_term)
                    #made_change = True

                    # Update model without the removed term
                    #ffit.fixall()
                    #setup_terms(ffit, selected_terms, initial_values)
                    #ffit.fit(fdata)
                    #best_wssrndf = ffit.wssrndf
                    # Next improvement should be calculated relative to this value
                    #last_stable_wssrndf = best_wssrndf


#        print(f"At the end: wssrndf={best_wssrndf}")
        # Stop if no changes were made in this iteration
        if not made_change:
            break

    # Final fit with all selected terms
    ffit.fixall()
    setup_terms(ffit, selected_terms, initial_values)
    ffit.fit(fdata)

    return selected_terms, ffit.wssrndf
