#!/usr/bin/env python3
"""
Stepwise regression module for photometric model fitting.
Includes functions for term expansion, model setup, and bidirectional stepwise regression.
"""

import numpy as np
import concurrent.futures
from copy import deepcopy

def format_polynomial_term(x_power, y_power):
    """
    Format polynomial term in human-friendly notation.

    Rules:
    - P0X1Y → PY (omit 0 coefficients, omit 1 exponents)
    - P1X0Y → PX
    - P1X1Y → PXY
    - P2X0Y → P2X
    - P0X2Y → P2Y
    - P2X3Y → P2X3Y

    Args:
        x_power (int): Power of X
        y_power (int): Power of Y

    Returns:
        str: Human-friendly term name
    """
    result = "P"

    # Handle X component
    if x_power > 0:
        if x_power == 1:
            result += "X"  # P1X → PX
        else:
            result += f"{x_power}X"  # P2X, P3X, etc.

    # Handle Y component
    if y_power > 0:
        if y_power == 1:
            result += "Y"  # P1Y → PY
        else:
            result += f"{y_power}Y"  # P2Y, P3Y, etc.

    return result

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
            # Surface polynomial - handle default order
            order_str = term[2:] if len(term) > 2 else '1'
            pol_order = int(order_str) if order_str else 1
            for pp in range(1, pol_order + 1):
                for rr in range(0, pp + 1):
                    term_name = format_polynomial_term(rr, pp-rr)
                    expanded_terms.append(term_name)
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
            # 1D polynomial (radial and other types) - handle default order
            order_str = term[2:] if len(term) > 2 else '1'
            pol_order = int(order_str) if order_str else 1
            for pp in range(1, pol_order + 1):
                if pp == 1:
                    # P1R → PR, P1C → PC, P1D → PD, etc. (omit the 1)
                    expanded_terms.append(f"P{term[1].upper()}")
                else:
                    # P2R, P3R, P2C, P3C, etc. (keep higher numbers)
                    expanded_terms.append(f"P{pp}{term[1].upper()}")
    else:
        # Regular term, no expansion needed
        expanded_terms.append(term)

    return expanded_terms

def parse_terms_simple(terms_string):
    """
    Parse the terms string into a list of individual terms (legacy function).

    Args:
        terms_string (str): Comma-separated string of terms.

    Returns:
        list: List of individual terms.
    """
    return [term.strip() for term in terms_string.split(',') if term.strip()]

def parse_terms(terms_string, n_images=1):
    """
    Parse extended term syntax supporting value assignment, fixed terms, and stepwise control.

    Syntax:
    - PC=0.2     : Set initial value
    - #PC=0.2    : Fix term at specific value
    - @PR        : Force stepwise regression
    - &P2X       : Force direct fitting
    - *PX        : Per-image term (expands to PX_1, PX_2, etc.)
    - PC         : Use default behavior based on --use-stepwise
    - .p3        : Macros work with all modifiers

    Args:
        terms_string (str): Comma-separated string of terms with modifiers
        n_images (int): Number of images for per-image term expansion

    Returns:
        dict: Dictionary with keys:
            - 'stepwise': list of terms for stepwise regression
            - 'direct': list of terms for direct fitting
            - 'fixed': dict of {term: value} for fixed terms
            - 'initial_values': dict of {term: value} for initial values
            - 'default': list of terms using default behavior
    """
    result = {
        'stepwise': [],
        'direct': [],
        'fixed': {},
        'initial_values': {},
        'default': []
    }

    if not terms_string:
        return result

    terms = [term.strip() for term in terms_string.split(',') if term.strip()]

    for term in terms:
        # Parse modifiers and values
        original_term = term
        is_fixed = False
        is_stepwise = False
        is_direct = False
        initial_value = None
        fixed_value = None

        # Parse all modifiers - they can be combined (e.g., *@PX, *#PC=0.1)
        is_per_image = False

        # Check for per-image modifier (*)
        if '*' in term:
            is_per_image = True
            term = term.replace('*', '', 1)  # Remove first occurrence

        # Check for fixed term modifier (#)
        if '#' in term:
            is_fixed = True
            term = term.replace('#', '', 1)  # Remove first occurrence

        # Check for stepwise modifier (@)
        elif '@' in term:
            is_stepwise = True
            term = term.replace('@', '', 1)  # Remove first occurrence

        # Check for direct modifier (&)
        elif '&' in term:
            is_direct = True
            term = term.replace('&', '', 1)  # Remove first occurrence

        # Check for value assignment (=)
        if '=' in term:
            term_name, value_str = term.split('=', 1)
            try:
                value = float(value_str)
                if is_fixed:
                    fixed_value = value
                else:
                    initial_value = value
            except ValueError:
                print(f"Warning: Invalid value '{value_str}' for term '{term_name}', ignoring")
                continue
            term = term_name

        # Expand pseudo-terms (e.g., .p3 -> P1X, PY, P1XY, etc.)
        expanded_terms = expand_pseudo_term(term)

        # Handle per-image expansion if * modifier was used
        if is_per_image:
            per_image_terms = []
            for base_term in expanded_terms:
                for img_idx in range(1, n_images + 1):
                    per_image_terms.append(f"{base_term}:{img_idx}")
            expanded_terms = per_image_terms

        # Apply the parsed modifiers to all expanded terms
        for expanded_term in expanded_terms:
            if is_fixed:
                # Fixed terms: use fixed_value if specified, otherwise need existing value
                result['fixed'][expanded_term] = fixed_value if fixed_value is not None else 0.0
            elif is_stepwise:
                result['stepwise'].append(expanded_term)
                if initial_value is not None:
                    result['initial_values'][expanded_term] = initial_value
            elif is_direct:
                result['direct'].append(expanded_term)
                if initial_value is not None:
                    result['initial_values'][expanded_term] = initial_value
            else:
                # Default behavior - will be determined by --use-stepwise setting
                result['default'].append(expanded_term)
                if initial_value is not None:
                    result['initial_values'][expanded_term] = initial_value

    # Conflict resolution with priorities:
    # 1. Fixed > everything else
    # 2. Per-image > global (for same base term)
    # 3. Direct > stepwise (for same scope)

    def resolve_conflicts(result):
        """Apply priority-based conflict resolution"""

        # Step 1: Fixed terms take precedence over everything
        fixed_terms = set(result['fixed'].keys())
        result['direct'] = [t for t in result['direct'] if t not in fixed_terms]
        result['stepwise'] = [t for t in result['stepwise'] if t not in fixed_terms]
        result['default'] = [t for t in result['default'] if t not in fixed_terms]

        # Step 2: Per-image terms remove conflicting global terms
        # Collect all per-image terms and their base names
        all_terms = result['direct'] + result['stepwise'] + result['default']
        per_image_bases = set()

        for term in all_terms:
            if ':' in term and term.split(':')[-1].isdigit():
                base_term = term.rsplit(':', 1)[0]
                per_image_bases.add(base_term)

        # Remove global versions of terms that have per-image versions
        def filter_global_conflicts(term_list):
            return [t for t in term_list if not (t in per_image_bases)]

        result['direct'] = filter_global_conflicts(result['direct'])
        result['stepwise'] = filter_global_conflicts(result['stepwise'])
        result['default'] = filter_global_conflicts(result['default'])

        # Step 3: Direct > stepwise > default (for same scope)
        direct_terms = set(result['direct'])
        result['stepwise'] = [t for t in result['stepwise'] if t not in direct_terms]
        result['default'] = [t for t in result['default'] if t not in direct_terms]

        stepwise_terms = set(result['stepwise'])
        result['default'] = [t for t in result['default'] if t not in stepwise_terms]

        return result

    result = resolve_conflicts(result)

    # Zeropoint terms are now explicitly added in dophot3.py with computed values
    # No need to auto-add them here

    return result

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

def perform_stepwise_regression(data, ffit, initial_terms, options, metadata, always_selected=None):
    """
    Perform bidirectional stepwise regression while maintaining robustness to outliers.
    Uses parallel processing for both forward and backward steps.

    Args:
        data: PhotometryData object containing the data
        ffit: fotfit object for model fitting
        initial_terms: List of initial terms to consider for stepwise selection
        options: Command line options
        metadata: List of metadata for the images
        always_selected: List of terms that should always be selected (never removed)

    Returns:
        tuple: (selected_terms, final_wssrndf)
    """
    # Initialize with always selected terms (direct terms)
    always_selected = always_selected or []
    selected_terms = always_selected.copy()
    remaining_terms = initial_terms.copy()

    # Get initial fit data
    fdata = data.get_arrays('y', 'adif', 'coord_x', 'coord_y', 'color1',
                           'color2', 'color3', 'color4', 'img', 'x', 'dy',
                           'image_x', 'image_y')

    # Initial fit with always selected terms
    ffit.fixall()
    if always_selected:
        # Set up always selected terms (direct terms) with their initial values
        initial_values_for_direct = {}
        if options.model:
            for term, value in zip(ffit.fixterms + ffit.fitterms,
                                 ffit.fixvalues + ffit.fitvalues):
                initial_values_for_direct[term] = value

        # Use command line initial values if available
        if hasattr(options, '_combined_initial_values'):
            initial_values_for_direct.update(options._combined_initial_values)

        values = [initial_values_for_direct.get(term, 1e-6) for term in always_selected]
        ffit.fitterm(always_selected, values=values)
        print(f"Starting with always-selected terms: {always_selected}")

    # Perform initial fit with sigma clipping using existing robust fitting function
    # This ensures photometry mask is always created, even when no stepwise terms are selected
    initial_wssrndf, initial_mask = try_term_robust(ffit, None, always_selected or [], fdata, initial_values_for_direct if 'initial_values_for_direct' in locals() else None)

    # Store the photometry mask in the data object for plotting
    data.add_mask('photometry', initial_mask)
    best_wssrndf = initial_wssrndf

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
            data.apply_mask(best_new_mask, 'photometry')

#            print(f"Before removal step: wssrndf={best_wssrndf}")
            # Backward step - check if any terms can be removed in parallel
            # Only consider removing terms that are not in always_selected
            removable_terms = [t for t in selected_terms if t not in always_selected]
            if len(removable_terms) > 0:  # Keep at least one term, but respect always_selected
                worst_term = None
                smallest_degradation = float('inf')

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    # Submit all term removal trials in parallel (only for removable terms)
                    futures = [
                        executor.submit(try_remove_term_parallel,
                                     term, selected_terms,
                                     ffit, fdata, initial_values)
                        for term in removable_terms
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
