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

        if proj is not None:
            try:
                self.fixterm(["PROJ"], [self.projections.index(proj)])
            except ValueError:
                print("Unsupported projection, defaulting to TAN")
                self.fixterm(["PROJ"], [self.PROJ_TAN])

        if file is not None:
            self.readmodel(file)

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
                arcsec = ", %.3f\""%(value*3600)
            output += "%-8s= %16.12f / ± %14.12f (%.6f%%%s)\n"%\
                (term, value, error, np.abs(100*error/np.sqrt(value*value+1e-10)), arcsec)
            i += 1
        output += "SIGMA   = %.3f\n"%(self.sigma)
        output += "WSSR/NDF= %.3f"%(self.wssrndf)

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
            print("Writing new WCS header into %s"%(file))
        else:
            print("%s does not exist, no WCS header written"%(file))
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
                fits.setval(file, 'CTYPE1', 0, value='RA---%3s'%self.projections[np.int64(value)])
                fits.setval(file, 'CTYPE2', 0, value='DEC--%3s'%self.projections[np.int64(value)])
            else:
                fits.setval(file, term, 0, value=value)

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
        x, y, ra, dec, err = data
        PV2 = [ 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0 ]

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
            if term == 'CRVAL1': CRVAL1 = value
            if term == 'CRVAL2': CRVAL2 = value
            if term == 'CRPIX1': CRPIX1 = value
            if term == 'CRPIX2': CRPIX2 = value

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
        #    if term == 'A': A = value
        #    if term == 'F': F = value

        # full trigonomotric transformation
        # from celestial (ra/dec) to image (u,b):
        u = np.arccos( np.sin(dec*rad)*np.sin(CRVAL2*rad)
            + np.cos(dec*rad)*np.cos(CRVAL2*rad)*np.cos((ra-CRVAL1)*rad) )
        b = np.arctan2( np.sin(dec*rad)*np.cos(CRVAL2*rad)
            - np.cos(dec*rad)*np.sin(CRVAL2*rad)*np.cos((ra-CRVAL1)*rad),
            -np.cos(dec*rad)*np.sin((ra-CRVAL1)*rad) )

        # the actual kernel of projection:
        if proj == self.PROJ_TAN:
            r = np.tan(u)

        if proj == self.PROJ_ZEA:
            r = np.sqrt(2*(1-np.sin(u)))

        if proj == self.PROJ_ZPN:
            r = 0
            for n in range(0,10):
                if PV2[n] != 0:
                    r += PV2[n] * np.power(u, n)

        if proj == self.PROJ_AZP:
            r = (PV2[1]+1)*np.sin(u)/(PV2[1]+np.cos(u))

        if proj == self.PROJ_SZP:
            xp = -PV2[1] * np.cos(PV2[3]*rad) * np.sin(PV2[2]*rad)
            yp =  PV2[1] * np.cos(PV2[3]*rad) * np.sin(PV2[2]*rad)
            zp =  PV2[1] * np.sin(PV2[3]*rad) + 1

            y1 = + (zp * np.cos(u)*np.sin(b) - xp * (1 - np.sin(u))) \
                    / (zp - (1 - np.sin(u))) / rad
            x1 = - (zp * np.cos(u)*np.cos(b) - xp * (1 - np.sin(u))) \
                    / (zp - (1 - np.sin(u))) / rad

        else:
            # in-image position in "degrees"
            y1 = + r * np.sin(b) / rad
            x1 = - r * np.cos(b) / rad

        # Apply the inverted linear transformation matrix CD:
        D = CD1_1 * CD2_2 - CD2_1 * CD1_2
        x = CRPIX1 + 1.0/D * (   CD2_2 * x1 - CD1_2 * y1)
        y = CRPIX2 + 1.0/D * ( - CD2_1 * x1 + CD1_1 * y1)

        return x,y

    def pixdist(self,x1,x2,y1,y2):
        """ distance in image
        x12, y12: image coordinates in pixels
        returns: distance in pixels """
        D = np.sqrt( np.power(x1-x2,2) + np.power(y1-y2,2))
        return D

    def residuals0(self, values, data):
        """pure residuals to compute sigma and similar things"""
        x, y, racat, deccat, err = data
        xmod,ymod = self.model(values, data)
        return self.pixdist(x,xmod,y,ymod)

    def residuals(self, values, data):
        """residuals for fitting with error weighting and delinearization"""
        x, y, racat, deccat, err = data
        xmod,ymod = self.model(values, data)
        dist = self.pixdist(x,xmod,y,ymod)/err
        if self.delin:
            return self.cauchy_delin(dist)
        else:
            return dist
