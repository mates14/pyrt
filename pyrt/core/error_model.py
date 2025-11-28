#!/usr/bin/python3
"""
error_model.py
Astrometric error model fitting
(c) Martin Jelinek, ASU AV CR, 2024
"""

import numpy as np
from pyrt.core import termfit
import logging

class ErrorModelFit(termfit.termfit):
    """
    Fit astrometric error model to post-fit residuals
    Model: σ²(centering_error, radius, ...) = S0 + SC*centering_error² + SR*r + ...
    Where:
    - S0: base systematic variance (pixels²)
    - SC: centering variance scaling (dimensionless)
    - SR: radial variance scaling (pixels/1000)
    """
    
    modelname = "Astrometric Error Model"
    
    def __init__(self):
        """Initialize error model fitter"""
        super().__init__()
        self.reference_magnitude = 15.0  # Will be set from data
        self.reference_radius = 1000.0   # Will be set from data
        self.reference_centering = 0.1   # Will be set from data
        
    def set_reference_values(self, magnitudes, radii):
        """Set reference values for centering the model"""
        self.reference_magnitude = np.median(magnitudes)
        self.reference_radius = np.median(radii)
        logging.info(f"Reference magnitude: {self.reference_magnitude:.2f}")
        logging.info(f"Reference radius: {self.reference_radius:.1f} pixels")
    
    def model(self, values, data):
        """
        Error variance model: σ²(centering_error, radius)
        
        Args:
            values: fitted parameter values
            data: (centering_err, radius, norm_centering, norm_radius, observed_variance)
        
        Returns:
            predicted_variance: Model prediction for σ²
        """
        centering_err, radius, norm_centering, norm_radius, observed_variance = data
        
        # Combine all parameters
        all_values = np.concatenate([values, np.array(self.fixvalues)])
        all_terms = self.fitterms + self.fixterms
        
        # Initialize model with base variance
        predicted_variance = np.zeros_like(centering_err)
        
        for term, value in zip(all_terms, all_values):
            if term == 'S0':  # Base variance (systematic floor)
                predicted_variance += value
            elif term == 'SC':  # Centering variance term (dimensionally correct)
                predicted_variance += value * norm_centering**2
            elif term == 'SC2':  # Quadratic centering error term  
                predicted_variance += value * norm_centering**2
            elif term == 'SR':  # Linear radius term
                predicted_variance += value * norm_radius
            elif term == 'SR2':  # Quadratic radius term
                predicted_variance += value * norm_radius**2
            elif term == 'SCR':  # Cross term centering*radius
                predicted_variance += value * norm_centering * norm_radius
        
        # Ensure positive variance
        return np.maximum(predicted_variance, 1e-10)
    
    def residuals(self, values, data):
        """
        Residuals for fitting error model
        
        We're fitting to squared residuals (variance), using chi-squared-like weighting
        """
        centering_err, radius, norm_centering, norm_radius, observed_variance = data
        
        predicted_variance = self.model(values, data)
        
        # Weight by inverse of expected standard deviation of variance estimate
        # For chi-squared distribution with 1 DOF: std(σ²) ≈ σ² * √(2)
        weights = 1.0 / np.sqrt(2 * predicted_variance + 1e-10)
        
        return weights * (observed_variance - predicted_variance)
    
    def residuals0(self, values, data):
        """Unweighted residuals for sigma computation"""
        centering_err, radius, norm_centering, norm_radius, observed_variance = data
        predicted_variance = self.model(values, data)
        return np.abs(observed_variance - predicted_variance)
    
    def fit_error_model_with_centering(self, astrometric_residuals, centering_errors, radii, initial_terms=None):
        """
        Fit error model to astrometric residuals using centering errors
        
        Args:
            astrometric_residuals: Post-fit astrometric residuals in pixels
            centering_errors: Individual star centering uncertainties from astrometric fit
            radii: Radial distance from image center in pixels
            initial_terms: List of terms to fit (default: ['S0', 'SC', 'SR'])
        
        Returns:
            Success flag
        """
        logging.info(f"fit_error_model_with_centering called with shapes: residuals={astrometric_residuals.shape}, centering={centering_errors.shape}, radii={radii.shape}")
        
        if initial_terms is None:
            initial_terms = ['S0', 'SC', 'SR']  # Base, centering error, radius terms
        
        # Set reference values for normalization
        self.reference_centering = np.median(centering_errors)
        self.reference_radius = np.median(radii)
        logging.info(f"Reference centering error: {self.reference_centering}")
        logging.info(f"Reference radius: {self.reference_radius:.1f} pixels")
        logging.info(f"Centering errors shape: {centering_errors.shape}, type: {type(centering_errors)}")
        logging.info(f"Sample centering errors: {centering_errors[:5] if len(centering_errors) > 5 else centering_errors}")
        
        # Normalize inputs for better numerical stability
        norm_centering = (centering_errors - self.reference_centering)
        norm_radius = (radii - self.reference_radius) / 1000.0  # Scale to ~1
        
        # Convert residuals to variance (square of residuals)
        observed_variance = astrometric_residuals**2
        
        # Prepare data tuple
        fit_data = (centering_errors, radii, norm_centering, norm_radius, observed_variance)
        
        # Save fitted data for debugging
        try:
            import os
            debug_filename = f"error_model_debug_{os.getpid()}.txt"
            with open(debug_filename, 'w') as f:
                f.write("# Error model fitting data\n")
                f.write(f"# N_points: {len(centering_errors)}\n")
                f.write(f"# Centering error stats: min={np.min(centering_errors):.6f}, max={np.max(centering_errors):.6f}, median={np.median(centering_errors):.6f}\n")
                f.write(f"# Astrometric residual stats: min={np.min(astrometric_residuals):.6f}, max={np.max(astrometric_residuals):.6f}, median={np.median(astrometric_residuals):.6f}\n")
                f.write(f"# Reference centering: {self.reference_centering:.6f}\n")
                f.write(f"# Reference radius: {self.reference_radius:.1f}\n")
                f.write("# centering_error norm_centering radius norm_radius observed_variance astrometric_residual\n")
                for i in range(len(centering_errors)):
                    f.write(f"{centering_errors[i]:.6f} {norm_centering[i]:.6f} {radii[i]:.1f} {norm_radius[i]:.6f} {observed_variance[i]:.6f} {astrometric_residuals[i]:.6f}\n")
            logging.info(f"Debug data saved to {debug_filename}")
        except Exception as e:
            logging.warning(f"Could not save debug data: {e}")
        
        # Set up initial parameter values
        initial_values = []
        for term in initial_terms:
            if term == 'S0':
                # Base variance - use median of observed variance
                initial_values.append(np.median(observed_variance))
            elif term == 'SC':
                # Centering variance scaling - dimensionless, expect values around 1-100
                initial_values.append(25.0)  # Typical value based on user experience
            elif term == 'SR':
                # Radius term - expect larger errors toward edges
                initial_values.append(0.01 * np.median(observed_variance))
            else:
                # Other terms start near zero
                initial_values.append(1e-6)
        
        # Set up the fit
        self.fitterm(initial_terms, initial_values)
        
        # Perform robust fit using binned medians instead of outlier rejection
        try:
            # Use robust binned median approach for scatter data
            n_points = len(centering_errors)
            bin_size = max(10, n_points // 19)  # At least 10 points per bin
            n_bins = n_points // bin_size
            
            logging.info(f"Using robust binned median fit: {n_points} points, {bin_size} per bin, {n_bins} bins")
            
            # Sort all data by centering error
            sort_indices = np.argsort(centering_errors)
            sorted_centering = centering_errors[sort_indices]
            sorted_observed_var = observed_variance[sort_indices]
            
            # Create bins and compute medians
            bin_centering_medians = []
            bin_variance_medians = []
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, n_points)
                
                bin_centering = sorted_centering[start_idx:end_idx]
                bin_variance = sorted_observed_var[start_idx:end_idx]
                
                # Compute medians for this bin
                median_centering = np.median(bin_centering)
                median_variance = np.median(bin_variance)
                
                bin_centering_medians.append(median_centering)
                bin_variance_medians.append(median_variance)
                
                logging.info(f"Bin {i+1}: centering={median_centering:.4f}, variance={median_variance:.4f} ({len(bin_centering)} points)")
            
            # Convert to arrays
            bin_centering_medians = np.array(bin_centering_medians)
            bin_variance_medians = np.array(bin_variance_medians)
            
            # Create normalized centering errors for binned data
            bin_norm_centering = bin_centering_medians - self.reference_centering
            bin_radii = np.ones_like(bin_centering_medians) * self.reference_radius  # Default radius
            bin_norm_radius = np.zeros_like(bin_centering_medians)  # No radial term for binned fit
            
            # Create fit data with binned medians
            binned_fit_data = (bin_centering_medians, bin_radii, bin_norm_centering, bin_norm_radius, bin_variance_medians)
            
            # Fit to the binned medians (robust approach)
            self.fit(binned_fit_data)
            
            # Check if SC coefficient is reasonable
            if 'SC' in initial_terms:
                sc_val = self.termval('SC')
                if abs(sc_val) > 1000:  # Unreasonable SC coefficient
                    logging.warning(f"SC coefficient ({sc_val:.1f}) is unreasonable, refitting with fixed SC=25")
                    
                    # Refit with fixed SC = 25 (typical value)
                    self.fixall()  # Fix all current terms
                    self.fitterm(['S0'], [np.median(bin_variance_medians)])  # Fit only S0
                    self.fixterm(['SC'], [25.0])  # Fix SC to reasonable value
                    
                    self.fit(binned_fit_data)
                    logging.info(f"Refitted with fixed SC=25")
            
            logging.info(f"Robust binned median fit completed successfully")
            logging.info(f"Final WSSR/NDF: {self.wssrndf:.3f}")
            return True
        except Exception as e:
            logging.error(f"Robust binned median fitting failed: {e}")
            return False
    
    def predict_error_with_centering(self, centering_error, radius):
        """
        Predict astrometric error (standard deviation) for given centering error and radius
        
        Args:
            centering_error: Individual star centering uncertainty
            radius: Radial distance from image center in pixels
        
        Returns:
            predicted_sigma: Predicted astrometric error in pixels
        """
        # Normalize inputs
        norm_centering = (centering_error - self.reference_centering)
        norm_radius = (radius - self.reference_radius) / 1000.0
        
        # Create dummy data for model evaluation
        dummy_variance = np.zeros_like(np.atleast_1d(centering_error))
        data = (np.atleast_1d(centering_error), np.atleast_1d(radius), 
                np.atleast_1d(norm_centering), np.atleast_1d(norm_radius), dummy_variance)
        
        # Get predicted variance
        predicted_variance = self.model(self.fitvalues, data)
        
        # Return standard deviation
        return np.sqrt(predicted_variance)
    
    def fit_error_model(self, astrometric_residuals, magnitudes, radii, initial_terms=None):
        """
        Fit error model to astrometric residuals
        
        Args:
            astrometric_residuals: Post-fit astrometric residuals in pixels
            magnitudes: Star magnitudes (MAG_AUTO)
            radii: Radial distance from image center in pixels
            initial_terms: List of terms to fit (default: ['S0', 'SM', 'SR'])
        
        Returns:
            Fitted error model
        """
        if initial_terms is None:
            initial_terms = ['S0', 'SM', 'SR']  # Base, magnitude, radius terms
        
        # Set reference values for normalization
        self.set_reference_values(magnitudes, radii)
        
        # Normalize inputs for better numerical stability
        norm_mag = (magnitudes - self.reference_magnitude)
        norm_radius = (radii - self.reference_radius) / 1000.0  # Scale to ~1
        
        # Convert residuals to variance (square of residuals)
        observed_variance = astrometric_residuals**2
        
        # Prepare data tuple
        fit_data = (magnitudes, radii, norm_mag, norm_radius, observed_variance)
        
        # Set up initial parameter values
        initial_values = []
        for term in initial_terms:
            if term == 'S0':
                # Base variance - use median of observed variance
                initial_values.append(np.median(observed_variance))
            elif term == 'SM':
                # Magnitude term - expect brighter stars to have smaller errors
                initial_values.append(-0.1 * np.median(observed_variance))
            elif term == 'SR':
                # Radius term - expect larger errors toward edges
                initial_values.append(0.01 * np.median(observed_variance))
            else:
                # Other terms start near zero
                initial_values.append(1e-6)
        
        # Set up the fit
        self.fitterm(initial_terms, initial_values)
        
        # Perform the fit
        try:
            self.fit(fit_data)
            
            # Check if SC coefficient is reasonable (for centering-based model)
            if 'SC' in initial_terms:
                sc_val = self.termval('SC')
                if abs(sc_val) > 1000:  # Unreasonable SC coefficient
                    logging.warning(f"SC coefficient ({sc_val:.1f}) is unreasonable, refitting with fixed SC=25")
                    
                    # Refit with fixed SC = 25 (typical value)
                    self.fixall()  # Fix all current terms
                    self.fitterm(['S0'], [np.median(observed_variance)])  # Fit only S0
                    self.fixterm(['SC'], [25.0])  # Fix SC to reasonable value
                    
                    self.fit(fit_data)
                    logging.info(f"Refitted with fixed SC=25")
            
            logging.info(f"Error model fitted successfully")
            logging.info(f"Final WSSR/NDF: {self.wssrndf:.3f}")
            return True
        except Exception as e:
            logging.error(f"Error model fitting failed: {e}")
            return False
    
    def predict_error(self, magnitude, radius):
        """
        Predict astrometric error (standard deviation) for given magnitude and radius
        
        Args:
            magnitude: Star magnitude
            radius: Radial distance from image center in pixels
        
        Returns:
            predicted_sigma: Predicted astrometric error in pixels
        """
        # Normalize inputs
        norm_mag = (magnitude - self.reference_magnitude)
        norm_radius = (radius - self.reference_radius) / 1000.0
        
        # Create dummy data for model evaluation
        dummy_variance = np.zeros_like(np.atleast_1d(magnitude))
        data = (np.atleast_1d(magnitude), np.atleast_1d(radius), 
                np.atleast_1d(norm_mag), np.atleast_1d(norm_radius), dummy_variance)
        
        # Get predicted variance
        predicted_variance = self.model(self.fitvalues, data)
        
        # Return standard deviation
        return np.sqrt(predicted_variance)
    
    def __str__(self):
        """Print error model parameters with physical interpretation"""
        output = super().__str__()
        
        # Add physical interpretation
        output += "\n\nPhysical Interpretation:\n"
        for term, value in zip(self.fitterms + self.fixterms, 
                              self.fitvalues + self.fixvalues):
            if term == 'S0':
                sigma0 = np.sqrt(abs(value))
                output += f"Base error (systematic): {sigma0:.4f} pixels\n"
            elif term == 'SC':
                output += f"Centering variance scaling: {value:.2f} (dimensionless)\n"
            elif term == 'SM':
                output += f"Magnitude dependence: {value:.6f} (negative = brighter stars have smaller errors)\n"
            elif term == 'SR':
                output += f"Radial dependence: {value:.6f} per 1000 pixels\n"
        
        return output