#!/usr/bin/python3
"""
fotfit.py
Photometric response fitter
(c) Martin Jelinek, ASU AV CR, 2023
"""

import numpy as np
import termfit
from astropy.table import Table

# wishlist :)
#from scipy.interpolate import RegularGridInterpolator

rad = np.pi/180.0

class fotfit(termfit.termfit):
    """Photometric response fitter
    """

    modelname = "FotFit photometric model"
    fit_xy = False
    zero = []
    px = []
    py = []

    def __init__(self, proj=None, file=None, fit_xy=False):
        """Start myself up"""

        super().__init__()

        self.fit_xy = fit_xy
        self.base_filter = None  # Store the filter used for fitting
        self.color_schema = None  # Store filters used for colors

        if file is not None:
            self.readmodel(file)
            self.fixall()

    def __str__(self):
        """Generate dynamic 2D table for per-image terms + global terms"""

        # Collect all per-image terms and organize by base term and image
        per_image_data = {}  # {base_term: {img_num: (value, error), ...}}
        max_img_num = 0

        # Process fitted terms
        for i, (term, value) in enumerate(zip(self.fitterms, self.fitvalues)):
            if ':' in term and term.split(':')[-1].isdigit():
                base_term = term.rsplit(':', 1)[0]
                img_num = int(term.split(':')[-1])
                max_img_num = max(max_img_num, img_num)

                try:
                    error = self.fiterrors[i] if hasattr(self, 'fiterrors') else np.nan
                except (IndexError, AttributeError):
                    error = np.nan

                if base_term not in per_image_data:
                    per_image_data[base_term] = {}
                per_image_data[base_term][img_num] = (value, error)

        # Process fixed terms
        for term, value in zip(self.fixterms, self.fixvalues):
            if ':' in term and term.split(':')[-1].isdigit():
                base_term = term.rsplit(':', 1)[0]
                img_num = int(term.split(':')[-1])
                max_img_num = max(max_img_num, img_num)

                if base_term not in per_image_data:
                    per_image_data[base_term] = {}
                per_image_data[base_term][img_num] = (value, 0.0)  # Fixed terms have 0 error

        # Generate 2D table if we have per-image terms
        output = ""
        if per_image_data:
            # Create header - prioritize Z first, then alphabetical
            base_terms = sorted(per_image_data.keys(), key=lambda x: (x != 'Z', x))

            # Build header
            header = " img "
            for base_term in base_terms:
                if base_term == 'Z':
                    header += "--zeropoint--- "
                else:
                    header += f"-----{base_term.lower()}------ "
            output += header + "\n"

            # Build data rows
            for img_num in range(1, max_img_num + 1):
                img_idx = img_num - 1  # Convert to 0-based for display
                row = f" {img_idx:03d} "

                for base_term in base_terms:
                    if img_num in per_image_data[base_term]:
                        value, error = per_image_data[base_term][img_num]
                        row += f"{value:6.3f} ± {error:5.3f} "
                    else:
                        row += "      -       "

                output += row + "\n"

        # Add global terms from parent class
        output += termfit.termfit.__str__(self)

        return output

    def set_filter_info(self, base_filter, color_schema):
        """Store filter information used in the fit"""
        self.base_filter = base_filter
        self.color_schema = color_schema

    def fit(self, data):

        self.fitvalues = self.fitvalues[0:(len(self.fitterms))] + self.zero
        if self.fit_xy:
            self.fitvalues = np.append(self.fitvalues, np.zeros(2*len(self.zero)))
        ret = termfit.termfit.fit(self, data)
        return ret

    def zero_val(self):
        """Extract zeropoint values and errors from Z:n terms

        Returns the stored zeropoint values directly (should be astronomical values ~25 mag).
        """
        zeropoint_values = []
        zeropoint_errors = []

        # Find Z:n terms in fitted terms
        for i, term in enumerate(self.fitterms):
            if term.startswith('Z:') and term.split(':')[-1].isdigit():
                zeropoint_values.append(self.fitvalues[i])
                try:
                    zeropoint_errors.append(self.fiterrors[i] if hasattr(self, 'fiterrors') else 0.0)
                except (IndexError, AttributeError):
                    zeropoint_errors.append(0.0)

        # Also check fixed terms for Z:n
        for i, term in enumerate(self.fixterms):
            if term.startswith('Z:') and term.split(':')[-1].isdigit():
                zeropoint_values.append(self.fixvalues[i])
                zeropoint_errors.append(0.0)  # Fixed terms have 0 error

        return np.array(zeropoint_values), np.array(zeropoint_errors)

    def mesh_flat(self, x, y, ctrx=512, ctry=512, img=0):
        """
        Generate flat field data efficiently for a given image using the existing model function.

        :param x: 2D array of x-coordinates
        :param y: 2D array of y-coordinates
        :param ctrx: x-coordinate of the image center
        :param ctry: y-coordinate of the image center
        :param img: image number
        :return: 2D array of flat field data
        """

        x_fine = x.astype(float) + 0.5  # Center of pixels
        y_fine = y.astype(float) + 0.5

        # Normalize coordinates
        coord_x = (x - ctrx) / 1024.0
        coord_y = (y - ctry) / 1024.0

        # Prepare the data array for the model function
        shape = x.shape
        data = (
            np.zeros(shape),  # mc (magnitude)
            np.ones(shape),   # airmass (set to 1 for flat field)
            coord_x,
            coord_y,
            np.zeros(shape),  # color1
            np.zeros(shape),  # color2
            np.zeros(shape),  # color3
            np.zeros(shape),  # color4
            np.full(shape, img),
            np.zeros(shape),  # y (not used in flat field calculation)
            np.ones(shape), # err (set to 1, not used in model calculation)
            x_fine, y_fine
            )

        # Use the existing model function to calculate the flat field
        return self.model(self.fitvalues, data)

    def old_flat(self, x, y, ctrx=512, ctry=512, img=0):
        return self.model(np.array(self.fitvalues), (0, 1, (x-ctrx)/1024., (y-ctry)/1024., 0, 0, 0, 0, img, 0, 0))

    def hyperbola(self, x, y):
        return (y*np.sqrt(1.0*x*x/y/y)+x)/2.0

    def model(self, values, data):
        """Optimized photometric response model"""
        mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err, cat_x, cat_y = data
        values = np.asarray(values)
        img = np.int64(img)

        radius2 = coord_x**2 + coord_y**2

        # Convert instrumental magnitude to counts
        counts = 10.0 ** (-0.4 * mc)

        # Transform counts to be relative to reference point
        REFERENCE_COUNTS = 10000.0
        OFFSET = 0.01
        transformed_counts = (counts + OFFSET) / REFERENCE_COUNTS
        mct = -2.5 * np.log10(transformed_counts)

        # Calculate base magnitude relative to reference point
        model = mct

        val2 = np.concatenate((values[0:len(self.fitterms)], np.array(self.fixvalues)))

        # Variables to collect rational function parameters
        rational_s = rational_o = rational_c = 0

        for term, value in zip(self.fitterms + self.fixterms, val2):
            # Check if this is a per-image term (has :n suffix)
            if ':' in term and term.split(':')[-1].isdigit():
                base_term = term.rsplit(':', 1)[0]
                term_img = int(term.split(':')[-1]) - 1  # Convert to 0-based indexing
                # Only apply this term to data points from the specified image
                img_mask = (img == term_img)
                if not np.any(img_mask):
                    continue  # Skip if no data points from this image
                term_to_process = base_term
            else:
                # Global term - apply to all images
                img_mask = np.ones_like(img, dtype=bool)
                term_to_process = term

            if term_to_process == 'Z':  # Zeropoint for this image
                # Apply 10 mag shift for numerical stability (astronomical zeropoint - 10)
                model[img_mask] += (value - 10.0)
            elif term_to_process == 'RS':  # Rational function parameter scale
                rational_s = value
            elif term_to_process == 'RO':  # Rational function parameter offset
                rational_o = value
            elif term_to_process == 'RC':  # Rational function parameter curvature
                rational_c = value
            elif term_to_process == 'SX':
                # Sinusoidal variation in X direction
                frac_x = cat_x - np.floor(cat_x)
                model[img_mask] += value * np.sin(np.pi * frac_x[img_mask])
            elif term_to_process == 'SY':
                # Sinusoidal variation in Y direction
                frac_y = cat_y - np.floor(cat_y)
                model[img_mask] += value * np.sin(np.pi * frac_y[img_mask])
            elif term_to_process == 'SXY':
                # Cross-term for X-Y sub-pixel variations
                frac_x = cat_x - np.floor(cat_x)
                frac_y = cat_y - np.floor(cat_y)
                model[img_mask] += value * np.sin(np.pi * frac_x[img_mask]) * np.sin(np.pi * frac_y[img_mask])
            elif term_to_process[0] == 'P':
                components = {'A': airmass, 'C': color1, 'D': color2, 'E': color3, 'F': color4,
                              'R': radius2, 'X': coord_x, 'Y': coord_y, 'N': mct }
                pterm = np.full_like(model, value)
                n = 1
                for a in term_to_process[1:]:
                    if a.isdigit():
                        n = int(a)
                    elif a in components:
                        pterm *= components[a]**n
                        n = 1
                model[img_mask] += pterm[img_mask]
            elif term_to_process == 'GA': ga = value
            elif term_to_process == 'GW': gw = 10**value
            elif term_to_process == 'EA': ea = value
            elif term_to_process == 'EW': ew = value
            elif term_to_process == 'EQ': eq = value
            elif term_to_process == 'HA': ha = value
            elif term_to_process == 'HM': hm = value
            elif term_to_process == 'HN': hn = value
            elif term_to_process == 'XC':
                bval = np.where(value < 0, value * color1,
                                np.where(value <= 1, value * color2,
                                         (value - 1) * color3 + color2))
                model[img_mask] += bval[img_mask]
            elif term_to_process == 'SC':
                s = np.sin(value * np.pi)
                w1 = 0.5 * (-s + 1)
                w2 = 0.5 * (s + 1)
                bval = np.where(value < -0.5, value * color1,
                                np.where(value < 0.5, w1 * value * color1 + w2 * value * color2,
                                         np.where(value <= 1.5, color2 + w2 * (value - 1) * color2 + w1 * (value - 1) * color3,
                                                  (value - 1) * color3 + color2)))
                model[img_mask] += bval[img_mask]

        if 'ga' in locals() and 'gw' in locals():
            model += ga * np.exp(-radius2 * 25 / gw)

        if 'ea' in locals() and 'ew' in locals():
            foo = np.sqrt(radius2 / ew)
            model += ea * (np.exp(foo) / (foo + 1) - 1)

        if any([rational_s, rational_o, rational_c]):
            x = -mct - rational_o  # Center the effect
            denom = 1 + rational_c * x * x  # Square term for stability
            rational = rational_s * x / (denom + 1e-10)  # Small epsilon to prevent division by zero
            model += rational
            #x = -mct
            #rational_correction = (rational_a + rational_b * x) / (1 + rational_c * x)
            #model += rational_correction

        # Zeropoints are now handled as regular Z:n terms in the main loop above
        # No need for special zeropoint handling here

        return model

    def isnumber(self, a):
        try:
            k=int(a)
            return True
        except:
            return False

    def residuals0(self, values, data):
        """pure residuals to compute sigma and similar things"""
        mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err, cat_x, cat_y = data
        return np.abs(y - self.model(values, data))

    def residuals(self, values, data):
        """residuals for fitting with error weighting and delinearization"""
        mc, airmass, coord_x, coord_y, color1, color2, color3, color4, img, y, err, cat_x, cat_y = data
        # Which power of the err is best here? Higher power prioritizes bright stars.
        dist = np.abs((y - self.model(values, data))/np.power(err,2.5))
        if self.delin:
            return self.cauchy_delin(dist)
        else:
            return dist

    def oneline(self):
        """Enhanced model string with filter information"""
        output=[]
        # Start with filter information
        if self.base_filter is not None:
            output.append(f"FILTER={self.base_filter}")
        if self.color_schema is not None:
            output.append(f"SCHEMA={self.color_schema}")

        # Add all fitted and fixed terms
        for term, value in zip(self.fixterms + self.fitterms,
                             self.fixvalues + self.fitvalues):
            output.append(f"{term}={value}")

        # If fit_xy is True, handle the px/py arrays
        if self.fit_xy:
            N = (len(self.fitvalues) - len(self.fitterms)) // 3
            px = self.fitvalues[len(self.fitterms) + N:len(self.fitterms) + 2*N]
            py = self.fitvalues[len(self.fitterms) + 2*N:len(self.fitterms) + 3*N]

            # Include only the PX/PY values for the current image
            # But need to know which image we're writing for...
            # current_img = det.meta.get('IMGNO', 0)  # This is tricky - how to get current image?
            output.append(f"P1X={px[0]}")
            output.append(f"P1Y={py[0]}")

        return ",".join(output)

    def oneline_for_image(self, img_idx):
        """Generate RESPONSE string for specific image, converting per-image terms to base names"""
        output = []

        # Start with filter information
        if self.base_filter is not None:
            output.append(f"FILTER={self.base_filter}")
        if self.color_schema is not None:
            output.append(f"SCHEMA={self.color_schema}")

        img_suffix = f":{img_idx}"

        # Process fitted and fixed terms
        for term, value in zip(self.fixterms + self.fitterms,
                             self.fixvalues + self.fitvalues):
            include_term = False
            output_term = term

            if term.endswith(img_suffix):
                # This is a per-image term for our image
                output_term = term.rsplit(':', 1)[0]  # Remove :n suffix
                include_term = True
            elif ':' not in term or not term.split(':')[-1].isdigit():
                # This is a global term (no :n or invalid :n)
                include_term = True

            if include_term:
                output.append(f"{output_term}={value:.6f}")

        return ",".join(output)

    def savemodel(self, file):
        """Enhanced model saving with complete metadata"""
        model_table = Table()

        # Add filter information
        model_table.meta['base_filter'] = self.base_filter
        model_table.meta['color_schema'] = self.color_schema
        model_table.meta['fit_xy'] = self.fit_xy

        # Add terms and values
        terms = []
        values = []
        errors = []

        # Add regular terms
        for term, value in zip(self.fitterms + self.fixterms,
                             self.fitvalues + self.fixvalues):
            terms.append(term)
            values.append(value)
            try:
                err = self.fiterrors[terms.index(term)] if term in self.fitterms else 0
            except IndexError:
                err = 0
            errors.append(err)

        # Add per-image PX/PY terms if fit_xy is True
        if self.fit_xy:
            N = (len(self.fitvalues) - len(self.fitterms)) // 3
            px = self.fitvalues[len(self.fitterms) + N:len(self.fitterms) + 2*N]
            py = self.fitvalues[len(self.fitterms) + 2*N:len(self.fitterms) + 3*N]
            for i, (px_val, py_val) in enumerate(zip(px, py)):
                terms.extend([f'PX{i}', f'PY{i}'])
                values.extend([px_val, py_val])
                errors.extend([0, 0])  # These are fitted but we don't save their errors

        model_table['term'] = terms
        model_table['val'] = values
        model_table['err'] = errors

        model_table.meta['name'] = self.modelname
        model_table.meta['sigma'] = self.sigma
        model_table.meta['variance'] = self.variance
        model_table.meta['wssrndf'] = self.wssrndf

        model_table.write(file, format='ascii.ecsv', overwrite=True)

    def readmodel(self, file):
        """Enhanced model reading with filter information"""
        model_table = Table.read(file, format='ascii.ecsv')

        # Read filter information
        self.base_filter = model_table.meta.get('base_filter')
        color_schema = model_table.meta.get('color_schema')
        #if color_schema is not None:
        #    self.color_schema = color_schema.split(',')
        self.fit_xy = model_table.meta.get('fit_xy', False)

        # Clear existing terms
        self.fixterms = []
        self.fixvalues = []
        self.fitterms = []
        self.fitvalues = []
        self.fiterrors = []

        # Read terms, handling PX/PY specially for fit_xy mode
        for param in model_table:
            term = param['term']
            if term.startswith(('PX', 'PY')) and self.fit_xy:
                # These will be handled by the fitting process
                continue
            elif param['err'] == 0:
                self.fixterms.append(term)
                self.fixvalues.append(param['val'])
            else:
                self.fitterms.append(term)
                self.fitvalues.append(param['val'])
                self.fiterrors.append(param['err'])

        # Read statistics
        #self.sigma = model_table.meta['sigma']
        #self.variance = model_table.meta['variance']
        #self.wssrndf = model_table.meta['wssrndf']

    def from_oneline(self, line):
        """
        Load model from a one-line string representation.

        Args:
            line (str): Model string in format "term1=value1,term2=value2,..."

        Example:
            >>> ffit = fotfit()
            >>> ffit.from_oneline("FILTER=Sloan_r,N1=1.0,PX=0.001,PY=-0.002")
        """
        # Clear existing terms
        self.fixterms = []
        self.fixvalues = []
        self.fitterms = []
        self.fitvalues = []
        self.fiterrors = []

        # Parse terms
        terms = line.split(',')
        img_px = {}  # Store PX terms by image number
        img_py = {}  # Store PY terms by image number

        for term in terms:
            try:
                name, strvalue = term.split('=')
                value = float(strvalue)

                # Handle filter information
                if name == 'FILTER':
                    self.base_filter = strvalue
                    continue
                elif name == 'SCHEMA':
                    self.color_schema = strvalue
                    continue

                # Add regular term
                self.fixterms.append(name)
                self.fixvalues.append(value)

            except ValueError as e:
                print(f"Warning: Could not parse term '{term}': {e}")
                continue

        return self

    def model_from_string(self, model_string, data):
        """
        Convenience method to load model from string and immediately apply it.

        Args:
            model_string (str): Model definition in one-line format
            data (tuple): Input data tuple for model evaluation

        Returns:
            array-like: Model results

        Example:
            >>> ffit = fotfit()
            >>> result = ffit.model_from_string("N1=1.0,PX=0.001", data_tuple)
        """
        self.from_oneline(model_string)
        return self.model(self.fixvalues, data)
