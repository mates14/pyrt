#!/usr/bin/python3
"""
termfit.py
generic fitter for models with terms
(c) Martin Jelinek, ASU AV CR, 2021-2023
"""

import numpy as np
import scipy.optimize as fit
from astropy.table import Table
import logging

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
            args=[data], ftol=1e-15)
        self.fitvalues = []
        for x in res.x:
            self.fitvalues += [ x ]
        # with two dimensional fit this does 2x the proper value (2x more data fitted)
        self.ndf = len(data[0]) - len(self.fitvalues)
        self.wssr = np.sum(self.residuals(self.fitvalues, data))
        self.sigma = np.median(self.residuals0(self.fitvalues, data)) / 0.67
        self.variance = np.median(self.residuals(self.fitvalues, data)) / 0.67
        self.wssrndf = self.wssr / self.ndf

        # Improved covariance matrix calculation with diagnostics
        try:
            # Method 1: Direct inverse (original method)
            jac_matrix = res.jac.T.dot(res.jac)

            # Check condition number
            cond_num = np.linalg.cond(jac_matrix)
#            logging.debug(f"Matrix condition number: {cond_num:.2e}")
#            if cond_num > 1e15:
#                logging.debug("Warning: Problem is ill-conditioned")

            try:
                cov = np.linalg.inv(jac_matrix)
#                logging.debug("Using direct inverse method")
                self.fiterrors = np.sqrt(np.abs(np.diagonal(cov)))
                return
            except np.linalg.LinAlgError:
                pass

            # Method 2: Use pseudo-inverse with SVD
            try:
                # Get the SVD components
                U, s, Vh = np.linalg.svd(jac_matrix)

                # Print singular values
#                logging.debug("\nSingular values:")
#                for i, sing_val in enumerate(s):
#                    logging.debug(f"σ_{i+1} = {sing_val:.2e}")

                # Calculate relative contributions
                rel_contributions = s / s[0]
#                logging.debug("\nRelative parameter contributions:")
#                for i, contrib in enumerate(rel_contributions):
#                    logging.debug(f"Parameter {i+1}: {contrib:.2e}")

                # Identify near-zero singular values (effectively rank-deficient)
                rank_threshold = 1e-12
                effective_rank = sum(s > rank_threshold * s[0])
#                logging.debug(f"\nEffective rank: {effective_rank} out of {len(s)}")

#                if effective_rank < len(s):
#                    logging.debug("Warning: Some parameters are effectively linearly dependent")
                    # Identify which parameters are problematic
#                    for i, (term, contrib) in enumerate(zip(self.fitterms, rel_contributions)):
#                        if contrib < rank_threshold:
#                            logging.debug(f"Parameter '{term}' may be linearly dependent with others")

                cov = np.linalg.pinv(jac_matrix, rcond=rank_threshold)
#                logging.debug("\nUsing SVD pseudo-inverse method")
                self.fiterrors = np.sqrt(np.abs(np.diagonal(cov)))

                # Check for zero or very small errors
                for i, (term, error) in enumerate(zip(self.fitterms, self.fiterrors)):
                    if error < 1e-10 * abs(self.fitvalues[i]):
#                        logging.debug(f"Warning: Parameter '{term}' has very small/zero error")
                        # Replace zero errors with a more meaningful estimate
                        self.fiterrors[i] = abs(self.fitvalues[i]) * 1e-6  # Conservative estimate

                return
            except Exception as e:
                logging.debug(f"SVD failed: {str(e)}")
                pass

            # Method 3: Add small regularization term
            try:
                epsilon = 1e-10 * np.trace(jac_matrix) / jac_matrix.shape[0]
                reg_matrix = jac_matrix + epsilon * np.eye(jac_matrix.shape[0])
                cov = np.linalg.inv(reg_matrix)
#                logging.debug("Using regularized inverse method")
                self.fiterrors = np.sqrt(np.abs(np.diagonal(cov)))
                return
            except:
                pass

            # Fallback: Estimate errors using parameter perturbation
#            logging.debug("Using parameter perturbation method")
            param_errors = []
            delta = 1e-6  # Small perturbation
            for i in range(len(self.fitvalues)):
                params_plus = self.fitvalues.copy()
                params_plus[i] += delta
                resid_plus = np.sum(self.residuals(params_plus, data)**2)

                params_minus = self.fitvalues.copy()
                params_minus[i] -= delta
                resid_minus = np.sum(self.residuals(params_minus, data)**2)

                # Estimate local curvature
                curvature = (resid_plus + resid_minus - 2*self.wssr) / (delta**2)
                if curvature > 0:
                    param_errors.append(np.sqrt(2.0/curvature))
                else:
                    param_errors.append(abs(self.fitvalues[i]) * 1e-6)  # Conservative estimate

            self.fiterrors = np.array(param_errors)

        except Exception as e:
            logging.debug(f"Warning: Error calculation failed with message: {str(e)}")
            self.fiterrors = res.x * np.nan

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

    def removeterm(self, term):
        """Remove a term from fitting terms or fixed terms"""
        # Remove from fitterms if present
        if term in self.fitterms:
            idx = self.fitterms.index(term)
            self.fitterms.pop(idx)
            self.fitvalues.pop(idx)
            if hasattr(self, 'fiterrors'):
                self.fiterrors.pop(idx)
        # Remove from fixterms if present
        elif term in self.fixterms:
            idx = self.fixterms.index(term)
            self.fixterms.pop(idx)
            self.fixvalues.pop(idx)
