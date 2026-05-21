#!/usr/bin/python3
"""
zpnfit.py
FITS/WCS/ZPN fitter
(c) Martin Jelinek, ASU AV CR, 2023
"""

import os
import numpy as np
#import scipy.optimize as fit
import scipy
#from astropy.table import Table
from astropy.io import fits
import astropy.wcs
import collections
from pyrt.core import termfit
import logging

rad = np.pi/180.0

class zpnfit(termfit.termfit):
    """WCS projection astrometry fitter
    Supports TAN, ZPN, ZEA, AZP, ARC projections
    """

    PROJ_TAN = 0
    PROJ_ZPN = 1
    PROJ_ZEA = 2
    PROJ_AZP = 3
    PROJ_SZP = 4
    PROJ_ARC = 5
    projections = [ "TAN", "ZPN", "ZEA", "AZP", "SZP", "ARC" ]
    print_in_arcsec = [ "CD1_1", "CD1_2", "CD2_1", "CD2_2" ]
    modelname="FITZPN astrometric model"

    def __init__(self, proj=None, file=None):
        """Start myself up"""
        super().__init__()  # Make sure to call the parent class initializer

        if proj is not None:
            try:
                self.fixterm(["PROJ"], [self.projections.index(proj)])
            except ValueError:
                logging.warning("Unsupported projection, defaulting to TAN")
                self.fixterm(["PROJ"], [self.PROJ_TAN])

        if file is not None:
            self.readmodel(file)

        self.sip_order = 0
        self.sip_a = np.zeros((self.sip_order + 1, self.sip_order + 1))
        self.sip_b = np.zeros((self.sip_order + 1, self.sip_order + 1))

    def __str__(self):
        """Print a WCS header contained in this class"""
        output = ""
        for term, value in zip(self.fixterms, self.fixvalues):
            display = self._display_term(term)
            if display == "PROJ":
                output += "%-10s= %16s / fixed\n" % (display, self.projections[np.int64(value)])
            else:
                output += "%-10s= %16f / fixed\n" % (display, value)

        i = 0
        for term, value in zip(self.fitterms, self.fitvalues):
            try: error = self.fiterrors[i]
            except IndexError: error = np.nan
            display = self._display_term(term)
            arcsec = ""
            base = term.rsplit(':', 1)[0] if ':' in term and term.rsplit(':', 1)[1].isdigit() else term
            if base in self.print_in_arcsec:
                arcsec = f", {value*3600:.3f}\""
            output += "%-10s= %16.12f / ± %14.12f (%.6f%%%s)\n" % \
                (display, value, error, np.abs(100*error/np.sqrt(value*value+1e-10)), arcsec)
            i += 1
        output += f"SIGMA   = {self.sigma:.3f}\n"
        output += f"WSSR/NDF= {self.wssrndf:.3f}"

        return output

    @staticmethod
    def _display_term(term):
        """Strip :0 suffix from term names for cleaner single-frame display."""
        if ':' in term:
            base, suffix = term.rsplit(':', 1)
            if suffix.isdigit() and int(suffix) == 0:
                return base
        return term

    def wcs(self, img_idx=0):
        """Return a FITS WCS header dict for the specified image index.

        Per-image terms (CRVAL1:n, CD1_1:n, etc.) are included only when n == img_idx
        and are written under their bare FITS keyword names (CRVAL1, CD1_1, …).
        Global terms (CRPIX1, CRPIX2, PV2_*, PROJ) are included for every image.
        """
        hdr = collections.OrderedDict()

        hdr['WCSAXES'] = 2  # must precede any other WCS keywords
        hdr['EQUINOX'] = 2000.0

        for term, value in zip(self.fixterms + self.fitterms,
                               self.fixvalues + self.fitvalues):
            fits_term = self._fits_term(term, img_idx)
            if fits_term is None:
                continue  # belongs to a different image
            if fits_term == "PROJ":
                hdr['CTYPE1'] = "RA---" + self.projections[np.int64(value)]
                hdr['CTYPE2'] = "DEC--" + self.projections[np.int64(value)]
                hdr['CUNIT1'] = "deg"
                hdr['CUNIT2'] = "deg"
            elif not fits_term.startswith(('A_', 'B_')):
                hdr[fits_term] = value

        if self.sip_order > 0:
            hdr['CTYPE1'] = hdr['CTYPE1'] + '-SIP'
            hdr['CTYPE2'] = hdr['CTYPE2'] + '-SIP'
            hdr['A_ORDER'] = self.sip_order
            hdr['B_ORDER'] = self.sip_order
            for i in range(self.sip_order + 1):
                for j in range(self.sip_order + 1 - i):
                    if i + j > 0:
                        hdr[f'A_{i}_{j}'] = self.sip_a[i, j]
                        hdr[f'B_{i}_{j}'] = self.sip_b[i, j]

        return hdr

    @staticmethod
    def _fits_term(term, img_idx):
        """Map an internal term name to its FITS keyword for the given image index.

        Returns the bare FITS keyword (stripping :n suffix) when the term belongs
        to img_idx, None when it belongs to a different image, or the term itself
        for global (non-per-image) terms.
        """
        if ':' in term:
            base, suffix = term.rsplit(':', 1)
            if suffix.isdigit():
                return base if int(suffix) == img_idx else None
        return term

    def add_sip_terms(self, order):
        """Add SIP terms with better initialization and scaling"""
        self.sip_order = order
        self.sip_a = np.zeros((order + 1, order + 1))
        self.sip_b = np.zeros((order + 1, order + 1))

        # Start with reasonable scales for different orders
        scales = {
            1: 1e-7,  # First order terms around 1e-6
            2: 1e-10, # Second order terms around 1e-9
            3: 1e-13, # Third order terms around 1e-12
            4: 1e-16  # Fourth order terms around 1e-15
        }

        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i + j > 0:  # Skip A_0_0 and B_0_0
                    scale = 1e-18  # Scale based on total order
#                    scale = scales.get(i + j, 1e-18)  # Scale based on total order
                    # Initialize with small random values around the expected scale
                    a_init = 0 # np.random.normal(0, scale)
                    b_init = 0 # np.random.normal(0, scale)
                    self.fitterm([f'A_{i}_{j}', f'B_{i}_{j}'], [a_init, b_init])

    def apply_sip(self, x, y):
        dx = np.sum(self.sip_a[i,j] * x**i * y**j
                    for i in range(self.sip_order + 1)
                    for j in range(self.sip_order + 1 - i)
                    if i + j > 0)
        dy = np.sum(self.sip_b[i,j] * x**i * y**j
                    for i in range(self.sip_order + 1)
                    for j in range(self.sip_order + 1 - i)
                    if i + j > 0)
        return x + dx, y + dy

    def write_wcs(self, filename, img_idx=0):
        """Write a minimal WCS-only FITS file (like astrometry.net .wcs output).

        Parameters
        ----------
        filename : str
            Output filename (typically with .wcs extension)
        img_idx : int
            Which image's per-image terms (CRVAL, CD matrix) to write (default 0)
        """
        # Create a minimal primary HDU with no data
        hdr = fits.Header()

        # Standard FITS header keywords
        hdr['SIMPLE'] = (True, 'Standard FITS file')
        hdr['BITPIX'] = (8, 'ASCII or bytes array')
        hdr['NAXIS'] = (0, 'Minimal header')
        hdr['EXTEND'] = (True, 'There may be FITS extensions')

        # Get WCS parameters from our model
        wcs_dict = self.wcs(img_idx)
        for key, value in wcs_dict.items():
            hdr[key] = value

        # Add provenance
        hdr['HISTORY'] = 'WCS solution created by pyrt/zpnfit'

        # Add quality metrics if available
        if hasattr(self, 'sigma') and not np.isnan(self.sigma):
            hdr['ASTSIGMA'] = (self.sigma, 'Astrometric WCS floor sqrt(S0) (pixels)')
        if hasattr(self, 'wssrndf') and not np.isnan(self.wssrndf):
            hdr['ASTWSSR'] = (self.wssrndf, 'Astrometric WSSR/NDF')
        if hasattr(self, 'scatter') and not np.isnan(self.scatter):
            hdr['ASTSCATT'] = (self.scatter, 'Astrometric scatter median(|res|)/0.67 (pixels)')

        # Create primary HDU with no data and write
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(filename, overwrite=True)
        logging.info(f"WCS solution written to {filename}")

    def write(self, output, clean_header=True, img_idx=0):
        """Write the fitted WCS solution to a FITS file or dictionary.

        Parameters
        ----------
        output : str or OrderedDict
            Either a path to a FITS file or an OrderedDict to store the WCS solution
        clean_header : bool, optional
            Whether to clean existing WCS-related keywords before writing new ones
        img_idx : int
            Which image's per-image terms (CRVAL, CD matrix) to write (default 0)

        Returns
        -------
        OrderedDict
            The updated header dictionary (only if output was an OrderedDict)
        """
        # Terms that should be cleaned up before writing new WCS
        wcs_terms = [
            # Basic WCS terms
            "WCSAXES", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
            "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2",
            "LONPOLE", "LATPOLE", "EQUINOX", "EPOCH", "RADESYS",
            # Projection-specific terms
            "PV1_0", "PV1_1", "PV1_2", "PV1_3", "PV1_4", "PV1_5",
            "PV2_0", "PV2_1", "PV2_2", "PV2_3", "PV2_4", "PV2_5",
            "PV2_6", "PV2_7", "PV2_8", "PV2_9", "PV2_10",
            # SIP terms
            "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"
        ]

        # Add SIP coefficient terms
        for i in range(5):  # Typical max order is 4
            for j in range(5-i):
                wcs_terms.extend([f'A_{i}_{j}', f'B_{i}_{j}', f'AP_{i}_{j}', f'BP_{i}_{j}'])

        # Function to write WCS keywords to either a FITS file or dictionary
        def write_wcs_keywords(target, is_file=True):
            # Clean existing WCS terms if requested
            if clean_header:
                if is_file:
                    for term in wcs_terms:
                        try:
                            fits.delval(target, term, 0)
                        except (KeyError, OSError):
                            pass
                    try:
                        fits.delval(target, "COMMENT", 0)
                    except (KeyError, OSError):
                        pass
                else:
                    for term in wcs_terms:
                        target.pop(term, None)

            # Write basic WCS terms
            def set_value(key, value):
                if is_file:
                    fits.setval(target, key, 0, value=value)
                else:
                    target[key] = value

            # Write WCSAXES first as required
            set_value('WCSAXES', 2)
            set_value('EQUINOX', 2000.0)

            # Write the solution sigma if available
            if hasattr(self, 'sigma'):
                try:
                    set_value('ASTSIGMA', self.sigma)
                except: # if it does not work, screw it, it is not mandatory
                    pass

            # Write astrometric solution quality parameters
            if hasattr(self, 'wssrndf'):
                try:
                    set_value('ASTWSSR', self.wssrndf)  # Astrometric WSSR/NDF
                except:
                    pass

            if hasattr(self, 'variance'):
                try:
                    set_value('ASTVAR', self.variance)  # Astrometric variance
                except:
                    pass

            if hasattr(self, 'scatter'):
                try:
                    set_value('ASTSCATT', self.scatter)  # Actual scatter: median(|res|)/0.67
                except:
                    pass

            # Write fixed and fitted terms
            for term, value in zip(self.fixterms + self.fitterms,
                                   self.fixvalues + self.fitvalues):
                fits_term = self._fits_term(term, img_idx)
                if fits_term is None:
                    continue  # belongs to a different image
                if fits_term == "PROJ":
                    proj = self.projections[int(value)]
                    suffix = "-SIP" if self.sip_order > 0 else ""
                    set_value('CTYPE1', f"RA---{proj}{suffix}")
                    set_value('CTYPE2', f"DEC--{proj}{suffix}")
                    set_value('CUNIT1', "deg")
                    set_value('CUNIT2', "deg")
                else:
                    if not fits_term.startswith(('A_', 'B_')):  # Defer SIP coefficients
                        set_value(fits_term, value)

            # Write SIP coefficients if present
            if self.sip_order > 0:
                set_value('A_ORDER', self.sip_order)
                set_value('B_ORDER', self.sip_order)

                for i in range(self.sip_order + 1):
                    for j in range(self.sip_order + 1 - i):
                        if i + j > 0:  # Skip A_0_0 and B_0_0
                            set_value(f'A_{i}_{j}', self.sip_a[i,j])
                            set_value(f'B_{i}_{j}', self.sip_b[i,j])

        # Handle different output types
        if isinstance(output, str):
            if not os.path.isfile(output):
                logging.warning(f"{output} does not exist, no WCS header written")
                return
            logging.info(f"Writing new WCS header into {output}")
            write_wcs_keywords(output, is_file=True)
        elif isinstance(output, dict):
            write_wcs_keywords(output, is_file=False)
            return output
        else:
            raise TypeError("Output must be either a file path (str) or dict")

    def model(self, values, data):
        """astrometric model inverted (fit in image plane)

        Global terms (shared across all images):
            CRPIX1, CRPIX2: centre of projection (optical axis)
            PV2_j: ZPN radial polynomial coefficients
            PROJ: projection type

        Per-image terms (CRVAL, CD matrix): stored with :n suffix for image n.
            CRVAL1:n, CRVAL2:n: sky coordinates at CRPIX for image n
            CD1_1:n, CD1_2:n, CD2_1:n, CD2_2:n: rotation/scale matrix for image n

        Bare names (CRVAL1, CD1_1, …) are also accepted for backward compatibility
        and treated as global (same value applied to all images).

        data: 5-tuple (image_x, image_y, ra, dec, image_dxy)          — single-frame
              6-tuple (image_x, image_y, ra, dec, image_dxy, img_idx) — multi-frame
        """
        x, y, ra, dec = data[0], data[1], data[2], data[3]
        # img_idx: integer array mapping each star to its image (0-based)
        if len(data) == 6:
            img_idx = np.asarray(data[5], dtype=int)
        else:
            img_idx = np.zeros(len(x), dtype=int)
        n_images = int(img_idx.max()) + 1 if len(img_idx) > 0 else 1

        PV2 = np.zeros(11)
        proj = self.PROJ_TAN
        CRPIX1 = 0.0
        CRPIX2 = 0.0

        # Per-image arrays (one slot per image)
        CRVAL1_arr = np.zeros(n_images)
        CRVAL2_arr = np.zeros(n_images)
        CD1_1_arr = np.ones(n_images)
        CD1_2_arr = np.zeros(n_images)
        CD2_1_arr = np.zeros(n_images)
        CD2_2_arr = np.ones(n_images)

        terms = self.fitterms + self.fixterms
        all_values = np.concatenate((values, np.array(self.fixvalues)))

        for term, value in zip(terms, all_values):
            if term == 'PROJ':   proj   = value
            if term == 'CRPIX1': CRPIX1 = value
            if term == 'CRPIX2': CRPIX2 = value
            if term == 'PV2_0':  PV2[0] = value
            if term == 'PV2_1':  PV2[1] = value
            if term == 'PV2_2':  PV2[2] = value
            if term == 'PV2_3':  PV2[3] = value
            if term == 'PV2_4':  PV2[4] = value
            if term == 'PV2_5':  PV2[5] = value
            if term == 'PV2_6':  PV2[6] = value
            if term == 'PV2_7':  PV2[7] = value
            if term == 'PV2_8':  PV2[8] = value
            if term == 'PV2_9':  PV2[9] = value
            if term == 'PV2_10': PV2[10] = value

            if self.sip_order > 0:
                for i in range(self.sip_order + 1):
                    for j in range(self.sip_order + 1 - i):
                        if i + j > 0:
                            if term == f'A_{i}_{j}': self.sip_a[i, j] = value
                            if term == f'B_{i}_{j}': self.sip_b[i, j] = value

            # Per-image terms: bare name → all images; name:n → specific image n
            for base, arr in (('CRVAL1', CRVAL1_arr), ('CRVAL2', CRVAL2_arr),
                               ('CD1_1', CD1_1_arr), ('CD1_2', CD1_2_arr),
                               ('CD2_1', CD2_1_arr), ('CD2_2', CD2_2_arr)):
                if term == base:
                    arr[:] = value
                elif term.startswith(base + ':') and term[len(base)+1:].isdigit():
                    n = int(term[len(base)+1:])
                    if n < n_images:
                        arr[n] = value

        # Dispatch per-star values via fancy indexing on img_idx
        CRVAL1 = CRVAL1_arr[img_idx]
        CRVAL2 = CRVAL2_arr[img_idx]
        CD1_1  = CD1_1_arr[img_idx]
        CD1_2  = CD1_2_arr[img_idx]
        CD2_1  = CD2_1_arr[img_idx]
        CD2_2  = CD2_2_arr[img_idx]

        # Handle spherical coordinate singularities
        ra = np.where(ra > 180, ra - 360, ra)
        dec = np.clip(dec, -90, 90)

        # Convert to radians
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        CRVAL1_rad = np.radians(CRVAL1)
        CRVAL2_rad = np.radians(CRVAL2)

        # Compute intermediate spherical coordinates
        cos_dec = np.cos(dec_rad)
        sin_dec = np.sin(dec_rad)
        cos_ra_diff = np.cos(ra_rad - CRVAL1_rad)

        u = np.arccos(sin_dec * np.sin(CRVAL2_rad) +
                      cos_dec * np.cos(CRVAL2_rad) * cos_ra_diff)

        b = np.arctan2(
            sin_dec * np.cos(CRVAL2_rad) - cos_dec * np.sin(CRVAL2_rad) * cos_ra_diff,
            -cos_dec * np.sin(ra_rad - CRVAL1_rad)
        )

        # the actual kernel of projection:
        if proj == self.PROJ_ZEA:
            # ZEA: r = sqrt(2*(1-sin(θ))) where θ is native latitude
            # Since u is angular distance from pole: θ = 90° - u, so sin(θ) = cos(u)
            r = np.sqrt(2*(1-np.cos(u)))
            x1 = -r * np.cos(b) / rad
            y1 = r * np.sin(b) / rad

        elif proj == self.PROJ_ZPN or proj == self.PROJ_ARC:
            # ZPN: zenithal polynomial with arbitrary radial function r(u)
            # ARC: zenithal equidistant (r = u), equivalent to ZPN with only PV2_1=1
            r = 0
            for n in range(0,10):
                if PV2[n] != 0:
                    r += PV2[n] * np.power(u, n)
            x1 = -r * np.cos(b) / rad
            y1 = r * np.sin(b) / rad

        elif proj == self.PROJ_AZP:
            # AZP: zenithal/azimuthal perspective projection
            # Parameters: mu (distance), gamma (tilt angle) in degrees
            mu = PV2[1]
            gamma = PV2[2]

            # Log field range once for diagnostic purposes
            if not hasattr(self, '_azp_field_debug_done'):
                u_deg = np.degrees(u)
                logging.debug(f"AZP: Field range u = {np.min(u_deg):.3f}° to {np.max(u_deg):.3f}° (angular distance from center)")
                self._azp_field_debug_done = True

            # Convert gamma to radians
            gamma_rad = gamma * rad

            # u is angular distance from pole, so theta (native latitude) = 90° - u
            # Therefore: sin(theta) = cos(u), cos(theta) = sin(u)
            # phi = b (position angle)

            # From wcstools azpfwd():
            # s = tan(gamma) * cos(phi)
            # t = (mu + sin(theta)) + cos(theta) * s
            # r = (mu+1) * cos(theta) / t
            # x = r * sin(phi)
            # y = -r * cos(phi) * sec(gamma)

            # Precompute trigonometric functions
            cos_gamma = np.cos(gamma_rad)
            sin_gamma = np.sin(gamma_rad)

            # Avoid division by zero for gamma
            if abs(cos_gamma) < 1e-10:
                cos_gamma = 1e-10

            tan_gamma = sin_gamma / cos_gamma
            sec_gamma = 1.0 / cos_gamma

            s = tan_gamma * np.cos(b)
            t = (mu + np.cos(u)) + np.sin(u) * s

            # Protect against division by zero
            t = np.where(np.abs(t) < 1e-10, 1e-10, t)

            r = (mu + 1.0) * np.sin(u) / t

            # Use same coordinate convention as TAN/ZEA/ZPN
            x1 = -r * np.cos(b) / rad
            y1 = r * np.sin(b) / rad

        elif proj == self.PROJ_SZP:
            # SZP: slant zenithal perspective projection
            # Parameters: mu (distance), phi_c (longitude), theta_c (latitude) in degrees
            mu = PV2[1]
            phi_c = PV2[2] * rad
            theta_c = PV2[3] * rad

            # Projection point coordinates
            xp = -mu * np.cos(theta_c) * np.sin(phi_c)
            yp = mu * np.cos(theta_c) * np.cos(phi_c)
            zp = mu * np.sin(theta_c) + 1

            # u is angular distance from pole, so theta (native latitude) = 90° - u
            # Therefore: sin(theta) = cos(u), cos(theta) = sin(u)
            s = 1.0 - np.cos(u)  # s = 1 - sin(theta)
            denominator = zp - s

            # Forward projection formulas from wcstools
            x1 = (zp * np.sin(u) * np.sin(b) - xp * s) / denominator / rad
            y1 = -(zp * np.sin(u) * np.cos(b) + yp * s) / denominator / rad

        else: # Default to TAN projection
            r = np.tan(u)
            x1 = -r * np.cos(b) / rad
            y1 = r * np.sin(b) / rad

        # Apply the inverted linear transformation matrix CD:
        D = CD1_1 * CD2_2 - CD2_1 * CD1_2
        x = 1.0/D * (   CD2_2 * x1 - CD1_2 * y1)
        y = 1.0/D * ( - CD2_1 * x1 + CD1_1 * y1)

        # Apply SIP distortion with proper normalization
        if True:
            if self.sip_order > 0:
                # Normalize coordinates to [-1,1] range for numerical stability
                x_center = (np.max(x) + np.min(x)) / 2
                y_center = (np.max(y) + np.min(y)) / 2
                x_scale = (np.max(x) - np.min(x)) / 2
                y_scale = (np.max(y) - np.min(y)) / 2

                x_norm = (x - x_center) / x_scale
                y_norm = (y - y_center) / y_scale

                dx = np.sum(self.sip_a[i,j] * x_norm**i * y_norm**j * (x_scale**i * y_scale**j)
                            for i in range(self.sip_order + 1)
                            for j in range(self.sip_order + 1 - i)
                            if i + j > 0)
                dy = np.sum(self.sip_b[i,j] * x_norm**i * y_norm**j * (x_scale**i * y_scale**j)
                            for i in range(self.sip_order + 1)
                            for j in range(self.sip_order + 1 - i)
                            if i + j > 0)
                x = x + dx
                y = y + dy

        return CRPIX1 + x, CRPIX2 + y

    def pixdist(self,x1,x2,y1,y2):
        """ distance in image
        x12, y12: image coordinates in pixels
        returns: distance in pixels """
        D = np.sqrt( np.power(x1-x2,2) + np.power(y1-y2,2))
        return D

    def residuals0(self, values, data):
        """pure residuals to compute sigma and similar things"""
        x, y = data[0], data[1]
        xmod, ymod = self.model(values, data)
        return self.pixdist(x, xmod, y, ymod)

    def residuals(self, values, data):
        """residuals for fitting with error weighting and delinearization"""
        x, y, err = data[0], data[1], data[4]
        xmod, ymod = self.model(values, data)
        dist = self.pixdist(x, xmod, y, ymod) / err
        if self.delin:
            return self.cauchy_delin(dist)
        return dist

    def calculate_ndf(self, data, stat_residuals):
        """Override NDF calculation for 2D astrometric fitting

        For astrometric fitting, each star provides 2 constraints (x,y)
        even though we return 1 residual per star
        """
        return 2 * len(stat_residuals) - len(self.fitvalues)
