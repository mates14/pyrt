#!/usr/bin/python3
"""
termfit.py
generic fitter for models with terms
(c) Martin Jelinek, ASU AV CR, 2021-2023
"""

import numpy as np
import scipy.optimize as fit
from astropy.table import Table

class termfit:
    """Fit data with string identified terms"""

    delin = False
    fitterms = []
    fitvalues = []
    fiterrors = []
    fixterms = []
    fixvalues = []
    sigma = np.nan
    variance = np.nan
    ndf = np.nan
    wssr = np.nan
    wssrndf = np.nan
    modelname="Model"

    def fixall(self):
        self.fixterms = self.fixterms + self.fitterms
        self.fixvalues = self.fixvalues + self.fitvalues
        self.fitterms = []
        self.fitvalues = []

    def fixterm(self, terms, values = None):
        """add and fixate a term"""

        if values is None:
            xvalues = []
            for term in terms:
                for ft,fv in zip(self.fixterms + self.fitterms, \
                    self.fixvalues + self.fitvalues):
                    if ft == term:
                        xvalues += [ fv ]
        else:
            xvalues = values

        for term,value in zip(terms,xvalues):
            newft = []
            newfv = []
            for ft,fv in zip(self.fitterms, self.fitvalues):
                if ft != term:
                    newft += [ ft ]
                    newfv += [ fv ]
            self.fitterms = newft
            self.fitvalues = newfv
            newft = []
            newfv = []
            for ft,fv in zip(self.fixterms, self.fixvalues):
                if ft != term:
                    newft += [ ft ]
                    newfv += [ fv ]
            self.fixterms = newft + [ term ]
            self.fixvalues = newfv + [ value ]

    def fitterm(self, terms, values = None):
        """add and set a term to be fitted"""
        if values is None:
            xvalues = []
            for term in terms:
                for ft,fv in zip(self.fixterms + self.fitterms, \
                    self.fixvalues + self.fitvalues):
                    if ft == term:
                        xvalues += [ fv ]
        else:
            xvalues = values

        for term,value in zip(terms,values):
            newft = []
            newfv = []
            for ft,fv in zip(self.fitterms, self.fitvalues):
                if ft != term:
                    newft += [ ft ]
                    newfv += [ fv ]
            self.fitterms = newft + [ term ]
            self.fitvalues = newfv + [ value ]
            newft = []
            newfv = []
            for ft,fv in zip(self.fixterms, self.fixvalues):
                if ft != term:
                    newft += [ ft ]
                    newfv += [ fv ]
            self.fixterms = newft
            self.fixvalues = newfv

    def termval(self, term):
        """return value of a term in question"""
        for ft,fv in zip(self.fixterms + self.fitterms, \
                            self.fixvalues + self.fitvalues):
            if ft == term:
                return fv
        # if one asks for a term not in the model
        return np.nan

    def __str__(self):
        """Print all terms fitted by this class"""
        output = ""
        for term,value in zip(self.fixterms,self.fixvalues):
            output += "%-8s= %16f / fixed\n"%(term,value)

        i=0
        for term, value in zip(self.fitterms, self.fitvalues):
            try: error = self.fiterrors[i]
            except IndexError: error = np.nan
            output += "%-8s= %16f / ± %f (%.3f%%)\n"%\
                (term, value, error, np.abs(100*error/value))
            i += 1
        output += "NDF     = %d\n"%(self.ndf)
        output += "SIGMA   = %.3f\n"%(self.sigma)
        output += "VARIANCE= %.3f\n"%(self.variance)
        output += "WSSR/NDF= %.3f"%(self.wssrndf)

        return output

    def oneline(self):
        """Print all terms fitted by this class in a single line that can be loaded later"""
        output = ""
        comma = False
        for term,value in zip(self.fixterms+self.fitterms,self.fixvalues+self.fitvalues):
            if comma: output += ","
            else: comma = True
            output += "%s=%f"%(term,value)
        return output

    def fit(self, data):
        """fit data to the defined model"""
        self.delin=False
        res = fit.least_squares(self.residuals, self.fitvalues,\
            args=[data], ftol=1e-14)
        self.fitvalues = []
        for x in res.x:
            self.fitvalues += [ x ]
        # with two dimensional fit this does 2x the proper value (2x more data fitted)
        self.ndf = len(data[0]) - len(self.fitvalues)
        self.wssr = np.sum(self.residuals(self.fitvalues, data))
        self.sigma = np.median(self.residuals0(self.fitvalues, data)) / 0.67
        self.variance = np.median(self.residuals(self.fitvalues, data)) / 0.67
        self.wssrndf = self.wssr / self.ndf

        try:
            cov = np.linalg.inv(res.jac.T.dot(res.jac))
            self.fiterrors = np.sqrt(np.diagonal(cov))
        except:
            self.fiterrors = res.x*np.nan

    def cauchy_delin(self, arg):
        """Cauchy delinearization to give outliers less weight and have
        more robust fitting"""
        try:
            ret=np.sqrt(np.log1p(arg**2))
        except RuntimeWarning:
            print(str(arg))
            ret=np.sqrt(np.log1p(arg**2))
        return ret

    def savemodel(self, file):
        """write model parameters into an ecsv file"""
        errs = []
        i=0
        for term in self.fitterms+self.fixterms:
            try:
                e = self.fiterrors[i]
            except IndexError: # fixed, not yet fitted etc.
                e = 0
            errs += [ e ]
            i+=1

        amodel = Table(
            [self.fitterms+self.fixterms,
            self.fitvalues[0:len(self.fitterms)]+self.fixvalues,errs],
            names=['term', 'val', 'err'])
        amodel.meta['name'] = self.modelname
        amodel.meta['sigma'] = self.sigma
        amodel.meta['variance'] = self.variance
        amodel.meta['wssrndf'] = self.wssrndf
        amodel.write(file, format="ascii.ecsv", overwrite=True)

    def readmodel(self, file):
        """Read model parameters from an ecsv file"""
        self.fixterms = []
        self.fixvalues = []
        self.fitterms = []
        self.fitvalues = []
        self.fiterrors = []

        rmodel = Table.read(file, format='ascii.ecsv')

        for param in rmodel:
            if param['err'] == 0:
                self.fixterms += [param['term']]
                self.fixvalues += [param['val']]
            else:
                self.fitterms += [param['term']]
                self.fitvalues += [param['val']]
                self.fiterrors += [param['err']]

        self.sigma = rmodel.meta['sigma']
        self.variance = rmodel.meta['variance']
        self.wssrndf = rmodel.meta['wssrndf']
