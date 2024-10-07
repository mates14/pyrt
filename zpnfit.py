#!/usr/bin/python3
"""
zpnfit.py
FITS/WCS/ZPN fitter
(c) Martin Jelinek, ASU AV CR, 2023
"""

import os
import numpy as np
#import scipy.optimize as fit
#from astropy.table import Table
from astropy.io import fits
import astropy.wcs
import collections
import termfit

import numpy as np

rad = np.pi/180.0

class zpnfit(termfit.termfit):
    """WCS/ZPN astrometry fitter
    Primarily made to fit ZPN, but can deal also with TAN,ZEA and AZP
    """

    PROJ_TAN = 0
    PROJ_ZPN = 1
    PROJ_ZEA = 2
    PROJ_AZP = 3
    PROJ_SZP = 4
    projections = [ "TAN", "ZPN", "ZEA", "AZP", "SZP" ]
    print_in_arcsec = [ "CD1_1", "CD1_2", "CD2_1", "CD2_2" ]
    modelname="FITZPN astrometric model"

    def __init__(self, proj=None, file=None):
        """Start myself up"""
        super().__init__()  # Make sure to call the parent class initializer

        if proj is not None:
            try:
                self.fixterm(["PROJ"], [self.projections.index(proj)])
            except ValueError:
                print("Unsupported projection, defaulting to TAN")
                self.fixterm(["PROJ"], [self.PROJ_TAN])

        if file is not None:
            self.readmodel(file)

        self.sip_order = 0
        self.sip_a = np.zeros((self.sip_order + 1, self.sip_order + 1))
        self.sip_b = np.zeros((self.sip_order + 1, self.sip_order + 1))

    def __str__(self):
        """Print a WCS header contained in this class"""
        output = ""
        for term,value in zip(self.fixterms,self.fixvalues):
            if term == "PROJ":
                output += "%-8s= %16s / fixed\n"%(term,self.projections[np.int64(value)])
            else:
                output += "%-8s= %16f / fixed\n"%(term,value)

        i=0
        for term, value in zip(self.fitterms, self.fitvalues):
            try: error = self.fiterrors[i]
            except IndexError: error = np.nan
            arcsec = ""
            if term in self.print_in_arcsec:
                arcsec = f", {value*3600:.3f}\""
            output += "%-8s= %16.12f / ± %14.12f (%.6f%%%s)\n"%\
                (term, value, error, np.abs(100*error/np.sqrt(value*value+1e-10)), arcsec)
            i += 1
        output += f"SIGMA   = {self.sigma:.3f}\n"
        output += f"WSSR/NDF= {self.wssrndf:.3f}"

        return output

    def wcs(self):
        """return a WCS header with a solution contained in self"""
        hdr = collections.OrderedDict()

        hdr['WCSAXES'] = 2  # must precede any other WCS keywords
        hdr['EQUINOX'] = 2000.0

        for term,value in zip(self.fixterms,self.fixvalues):
            if term == "PROJ":
                hdr['CTYPE1'] = "RA---"+self.projections[np.int64(value)]
                hdr['CTYPE2'] = "DEC--"+self.projections[np.int64(value)]
                hdr['CUNIT1'] = "deg"
                hdr['CUNIT2'] = "deg"
            else:
                hdr[term] = value

        for term, value in zip(self.fitterms, self.fitvalues):
            hdr[term] = value

        return hdr

    def add_sip_terms(self, order):
        self.sip_order = order
        self.sip_a = np.zeros((order + 1, order + 1))
        self.sip_b = np.zeros((order + 1, order + 1))
        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i + j > 0:  # Skip A_0_0 and B_0_0
                    self.fitterm([f'A_{i}_{j}', f'B_{i}_{j}'], [0, 0])

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

    def write(self, file):
        """ Write the fitted WCS solution to a file"""
        fuj_terms = [
            "A_ORDER", "A_0_0", "A_0_1", "A_0_2", "A_0_3", "A_0_4", "A_1_0",
            "A_1_1", "A_1_2", "A_1_3", "A_2_0", "A_2_1", "A_2_2", "A_3_0",
            "A_3_1", "A_4_0", "B_ORDER", "B_0_0", "B_0_1", "B_0_2", "B_0_3",
            "B_0_4", "B_1_0", "B_1_1", "B_1_2", "B_1_3", "B_2_0", "B_2_1",
            "B_2_2", "B_3_0", "B_3_1", "B_4_0", "AP_ORDER", "AP_0_0",
            "AP_0_1", "AP_0_2", "AP_0_3", "AP_0_4", "AP_1_0", "AP_1_1",
            "AP_1_2", "AP_1_3", "AP_2_0", "AP_2_1", "AP_2_2", "AP_3_0",
            "AP_3_1", "AP_4_0", "BP_ORDER", "BP_0_0", "BP_0_1", "BP_0_2",
            "BP_0_3", "BP_0_4", "BP_1_0", "BP_1_1", "BP_1_2", "BP_1_3",
            "BP_2_0", "BP_2_1", "BP_2_2", "BP_3_0", "BP_3_1", "BP_4_0" ]

        if os.path.isfile(file):
            print(f"Writing new WCS header into {file}")
        else:
            print(f"{file} does not exist, no WCS header written")
            return

        nic=0
        for term in fuj_terms:
#            os.system("fitsheader -d %s %s"%(term, file))
            try:
                fits.delval(file, term, 0)
            except:
                nic+=1

        try:
            fits.delval(file, "COMMENT", 0)
        except:
            nic+=1

        fits.setval(file, "ASTSIGMA", 0, value=self.sigma)

        for term, value in zip(
                self.fixterms + self.fitterms,
                self.fixvalues + self.fitvalues):
            if term == "PROJ":
                # And if SIP, change to --SIP?
                fits.setval(file, 'CTYPE1', 0, value=f"RA---{self.projections[np.int64(value)]:3s}")
                fits.setval(file, 'CTYPE2', 0, value=f"DEC--{self.projections[np.int64(value)]:3s}")
            else:
                fits.setval(file, term, 0, value=value)

        # Add SIP keywords to the FITS header
        if self.sip_order > 0:
            fits.setval(file, 'A_ORDER', 0, value=self.sip_order)
            fits.setval(file, 'B_ORDER', 0, value=self.sip_order)

            for i in range(self.sip_order + 1):
                for j in range(self.sip_order + 1 - i):
                    if i + j > 0:  # Skip A_0_0 and B_0_0
                        fits.setval(file, f'A_{i}_{j}', 0, value=self.sip_a[i,j])
                        fits.setval(file, f'B_{i}_{j}', 0, value=self.sip_b[i,j])

            # Update CTYPE to include SIP
            fits.setval(file, 'CTYPE1', 0, value=f"RA---{self.projections[np.int64(self.termval('PROJ'))]:3s}-SIP")
            fits.setval(file, 'CTYPE2', 0, value=f"DEC--{self.projections[np.int64(self.termval('PROJ'))]:3s}-SIP")

    def model(self, values, data):
        """astrometric model inverted (fit in image plane)

        fittable terms:
        CRPIX1,CRPIX2: center of projection
        CRVAL1,CRVAL2: sky coordinates at CRPIX1,CRPIX2
        CDi_j: rotation matrix in image plane
        PV2_j: for ZPN polynomial indices of the radial fit
               for AZP PV2_1 is mu value (see Calabretta)
        experimental:
        A,F: Amplitude and phase of projection tilt
        """
        x, y, ra, dec, _ = data # _ is err
        PV2 = np.zeros(11)  # Initialize PV2 array

        terms = self.fitterms + self.fixterms
        values = np.concatenate((values, np.array(self.fixvalues)))

        for term,value in zip(terms, values):

            # TAN, ZPN, ZEA and AZP supported
            if term == 'PROJ': proj = value

            # std terms of WCS model:
            if term == 'CD1_1': CD1_1 = value
            if term == 'CD1_2': CD1_2 = value
            if term == 'CD2_1': CD2_1 = value
            if term == 'CD2_2': CD2_2 = value
            if term == 'CRPIX1': CRPIX1 = value
            if term == 'CRPIX2': CRPIX2 = value
            # TODO: if this is to wrk with more images simultaneously,
            # these CRVALn terms need to be in an array
            if term == 'CRVAL1': CRVAL1 = value
            if term == 'CRVAL2': CRVAL2 = value

            # some extra
            if term == 'PV2_0': PV2[0] = value
            if term == 'PV2_1': PV2[1] = value
            if term == 'PV2_2': PV2[2] = value
            if term == 'PV2_3': PV2[3] = value
            if term == 'PV2_4': PV2[4] = value
            if term == 'PV2_5': PV2[5] = value
            if term == 'PV2_6': PV2[6] = value
            if term == 'PV2_7': PV2[7] = value
            if term == 'PV2_8': PV2[8] = value
            if term == 'PV2_9': PV2[9] = value
            if term == 'PV2_10': PV2[10] = value

            if self.sip_order > 0:
                for i in range(self.sip_order + 1):
                    for j in range(self.sip_order + 1 - i):
                        if i + j > 0:  # Skip A_0_0 and B_0_0
                            if term == f'A_{i}_{j}': self.sip_a[i,j] = value
                            if term == f'B_{i}_{j}': self.sip_b[i,j] = value

        #    if term == 'A': A = value
        #    if term == 'F': F = value

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
            r = np.sqrt(2*(1-np.sin(u)))
            x1 = -r * np.cos(b) / rad
            y1 = r * np.sin(b) / rad

        elif proj == self.PROJ_ZPN:
            r = 0
            for n in range(0,10):
                if PV2[n] != 0:
                    r += PV2[n] * np.power(u, n)
            x1 = -r * np.cos(b) / rad
            y1 = r * np.sin(b) / rad

        elif proj == self.PROJ_AZP:
            mu = PV2[1]
            gamma = PV2[2] if len(PV2) > 2 else 0.0  # Default gamma to 0 if not provided

            # Convert gamma to radians
            gamma_rad = gamma * np.pi / 180.0

            # Calculate intermediate values
            rho = np.sqrt((mu + 1)**2 - (mu + np.cos(u))**2)
            omega = mu + np.cos(u)

            # Calculate x and y
            x1 = -rho * np.sin(b) / (omega * np.cos(gamma_rad) - rho * np.sin(gamma_rad))
            y1 = rho * (omega * np.sin(gamma_rad) + rho * np.cos(gamma_rad)) / \
                 (omega * np.cos(gamma_rad) - rho * np.sin(gamma_rad)) / np.cos(b)

#            r = (PV2[1]+1)*np.sin(u)/(PV2[1]+np.cos(u))
#            x1 = -r * np.cos(b) / rad
#            y1 = r * np.sin(b) / rad

        elif proj == self.PROJ_SZP:
            mu = PV2[1]
            phi0 = PV2[2] * rad
            theta0 = PV2[3] * rad

            x = -mu * np.cos(theta0) * np.sin(phi0)
            y = mu * np.cos(theta0) * np.cos(phi0)
            z = mu * np.sin(theta0) + 1

            denominator = z - (1 - np.sin(u))
            x1 = -(z * np.cos(u) * np.cos(b) - x * (1 - np.sin(u))) / denominator / rad
            y1 = (z * np.cos(u) * np.sin(b) - y * (1 - np.sin(u))) / denominator / rad

        else: # Default to TAN projection
            r = np.tan(u)
            x1 = -r * np.cos(b) / rad
            y1 = r * np.sin(b) / rad

        # Apply the inverted linear transformation matrix CD:
        D = CD1_1 * CD2_2 - CD2_1 * CD1_2
        x = 1.0/D * (   CD2_2 * x1 - CD1_2 * y1)
        y = 1.0/D * ( - CD2_1 * x1 + CD1_1 * y1)

        # Apply SIP distortion
        if self.sip_order > 0:
            dx = np.sum(self.sip_a[i,j] * x**i * y**j
                        for i in range(self.sip_order + 1)
                        for j in range(self.sip_order + 1 - i)
                        if i + j > 0)
            dy = np.sum(self.sip_b[i,j] * x**i * y**j
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
        x, y, _, _, _ = data
        xmod,ymod = self.model(values, data)
        return self.pixdist(x,xmod,y,ymod)

    def residuals(self, values, data):
        """residuals for fitting with error weighting and delinearization"""
        x, y, _, _, err = data
        xmod,ymod = self.model(values, data)
        dist = self.pixdist(x,xmod,y,ymod)/err
        if self.delin:
            return self.cauchy_delin(dist)
        return dist
