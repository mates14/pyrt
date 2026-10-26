#!/usr/bin/python3
"""
termfit.py
generic fitter for models with terms
(c) Martin Jelinek, ASU AV CR, 2021-2023
"""

import os
import numpy as np
import scipy.optimize as fit
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import cho_factor, cho_solve
from astropy.table import Table
import logging

# 'linear': fits whose terms are all linear are solved directly (default),
# 'lsq': least_squares for every fit (the previous behaviour, for comparisons)
SOLVER = os.environ.get('PYRT_SOLVER', 'linear')

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
    varpro = False

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
        """Print all terms fitted by this class, excluding per-image terms (handled by fotfit)"""
        output = ""

        # Show fixed terms (excluding per-image)
        for term, value in zip(self.fixterms, self.fixvalues):
            if ':' in term and term.split(':')[-1].isdigit():
                continue  # Skip per-image terms
            output += "%-8s= %16f / fixed\n" % (term, value)

        # Show fitted terms (excluding per-image)
        i = 0
        for term, value in zip(self.fitterms, self.fitvalues):
            if ':' in term and term.split(':')[-1].isdigit():
                i += 1  # Still increment index to keep errors aligned
                continue  # Skip per-image terms

            try:
                error = self.fiterrors[i]
            except (IndexError, AttributeError):
                error = np.nan
            output += "%-8s= %16f / ± %f (%.3f%%)\n" % \
                (term, value, error, np.abs(100*error/value))
            i += 1

        output += "NDF     = %d\n" % (self.ndf)
        output += "SIGMA   = %.3f\n" % (self.sigma)
        output += "VARIANCE= %.3f\n" % (self.variance)
        output += "WSSR/NDF= %.3f" % (self.wssrndf)

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

    def oneline_for_image(self, img_idx):
        """Generate RESPONSE string for specific image, converting per-image terms to base names"""
        output = ""
        comma = False
        img_suffix = f":{img_idx}"

        for term, value in zip(self.fixterms + self.fitterms, self.fixvalues + self.fitvalues):
            include_term = False
            output_term = term

            if term.endswith(img_suffix):
                # This is a per-image term for our image - convert PX:1 → PX, Z:1 → Z
                output_term = term.rsplit(':', 1)[0]
                include_term = True
            elif ':' not in term or not term.split(':')[-1].isdigit():
                # This is a global term (no :n suffix) - include for all images
                include_term = True
            # Terms with other image suffixes are excluded

            if include_term:
                if comma:
                    output += ","
                else:
                    comma = True
                output += f"{output_term}={value:f}"

        return output

    def fit_residuals(self, values, data):
        """Residuals used for fitting - defaults to residuals() method"""
        return self.residuals(values, data)

    def calculate_ndf(self, data, stat_residuals):
        """Calculate degrees of freedom - can be overridden by subclasses

        Default implementation assumes 1D fitting where each residual
        represents one constraint
        """
        return len(stat_residuals) - len(self.fitvalues)

    def term_bounds(self, terms):
        """Per-term (lower, upper) bounds for fitting. Default: unbounded.

        Override in a subclass to constrain specific term types (e.g. fotfit
        restricts F<n> flux-weight terms to [0,1], since they have no
        physical meaning outside that range).
        """
        return [(-np.inf, np.inf) for _ in terms]

    def fit(self, data):
        """fit data to the defined model"""
        # Do NOT reset delin here — caller (e.g. refine_fit) may have enabled Cauchy
        # Minimize fit_residuals (may differ from residuals(), e.g. Cauchy delin)
        bounds = self.term_bounds(self.fitterms)
        lower = [b[0] for b in bounds]
        upper = [b[1] for b in bounds]
        # An initial guess can come from a previous (possibly unconstrained,
        # e.g. pre-existing RESPONSE header) fit and may violate the current
        # bounds; least_squares raises immediately if so rather than just
        # projecting it in, so clamp explicitly.
        clamped = np.clip(self.fitvalues, lower, upper)
        for term, orig, new in zip(self.fitterms, self.fitvalues, clamped):
            if orig != new:
                logging.warning(f"Initial value for '{term}' ({orig}) outside bounds, "
                                 f"clamped to {new}")
        self.fitvalues = list(clamped)
        lin, non = self._split_linear(bounds)
        if not self.fitterms:
            # Nothing free (e.g. -S0 tries a SIP of order 0 on a fixed model):
            # only the statistics of the model as it is.  scipy < 1.13
            # least_squares fails on an empty x0 (zero-size array, max)
            values, jac = [], None
        elif SOLVER == 'linear' and lin and not non:
            values, jac = self._fit_linear(data, lin)
        elif SOLVER == 'linear' and lin and self.varpro:
            values, jac = self._fit_varpro(data, lin, non, lower, upper)
        else:
            res = fit.least_squares(self.fit_residuals, self.fitvalues,\
                args=[data], ftol=1e-15, x_scale='jac', bounds=(lower, upper))
            values, jac = res.x, res.jac
        self.fitvalues = []
        for x in values:
            self.fitvalues += [ x ]
        # Compute statistics with plain residuals regardless of delin state
        saved_delin = self.delin
        self.delin = False
        stat_residuals = self.residuals(self.fitvalues, data)
        self.delin = saved_delin

        # Calculate degrees of freedom
        self.ndf = self.calculate_ndf(data, stat_residuals)
        self.wssr = np.sum(stat_residuals**2)  # Proper weighted sum of squares
        self.variance = np.std(stat_residuals)  # Standard deviation of normalized residuals

        # Sigma from unweighted residuals (measure of absolute scatter)
        self.sigma = np.median(self.residuals0(self.fitvalues, data)) / 0.67
        self.wssrndf = self.wssr / self.ndf

        # Improved covariance matrix calculation with diagnostics
        self.fitcov = None   # full covariance where available (zero_val propagates it)
        if jac is None:
            self.fitcov = np.zeros((0, 0))
            self.fiterrors = np.array([])
            return
        try:
            # Columns scaled to unit norm before inverting, the covariance
            # unscaled after: terms can differ by many orders of magnitude in
            # scale (SIP u^5 against CRPIX), which J^T J itself cannot survive
            colscale = np.sqrt(np.sum(np.asarray(jac)**2, axis=0))
            colscale[colscale == 0] = 1.0
            unscale = np.outer(colscale, colscale)
            jacs = jac / colscale

            # Method 1: Direct inverse (original method)
            jac_matrix = jacs.T.dot(jacs)

            # Check condition number
            cond_num = np.linalg.cond(jac_matrix)

            try:
                cov = np.linalg.inv(jac_matrix) / unscale
#                logging.debug("Using direct inverse method")
                self.fitcov = cov
                self.fiterrors = np.sqrt(np.abs(np.diagonal(cov)))
                return
            except np.linalg.LinAlgError:
                pass

            # Method 2: Use pseudo-inverse with SVD
            try:
                # Get the SVD components
                U, s, Vh = np.linalg.svd(jac_matrix)

                # Calculate relative contributions
                rel_contributions = s / s[0]

                # Identify near-zero singular values (effectively rank-deficient)
                rank_threshold = 1e-12
                effective_rank = sum(s > rank_threshold * s[0])
                cov = np.linalg.pinv(jac_matrix, rcond=rank_threshold) / unscale
#                logging.debug("Using SVD pseudo-inverse method")
                self.fitcov = cov
                self.fiterrors = np.sqrt(np.abs(np.diagonal(cov)))

                # Check for zero or very small errors
                for i, (term, error) in enumerate(zip(self.fitterms, self.fiterrors)):
                    if error < 1e-10 * abs(self.fitvalues[i]):
                        # Replace zero errors with a more meaningful estimate
                        self.fiterrors[i] = abs(self.fitvalues[i]) * 1e-6  # Conservative estimate

                return
            except Exception as e:
 #               logging.debug(f"SVD failed: {str(e)}")
                pass

            # Method 3: Add small regularization term
            try:
                epsilon = 1e-10 * np.trace(jac_matrix) / jac_matrix.shape[0]
                reg_matrix = jac_matrix + epsilon * np.eye(jac_matrix.shape[0])
                cov = np.linalg.inv(reg_matrix) / unscale
#                logging.debug("Using regularized inverse method")
                self.fitcov = cov
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
            self.fiterrors = np.asarray(values) * np.nan

    # --- direct linear solve ---------------------------------------------
    #
    # Most models are linear in most of their terms (zeropoint, polynomials,
    # SIP, CRPIX).  When every fitted term is linear the fit is one weighted
    # linear least-squares solve instead of an iterative search, exact and
    # independent of the starting values; least_squares stops early from a
    # cold start when the terms are tiny (SIP).  Cauchy delinearization is
    # reproduced by iteratively reweighted least squares with weights
    # 1/(1+d^2), which minimizes the same sum of log(1+d^2).  Fits with any
    # nonlinear term keep using least_squares over all terms, unless the class
    # sets varpro: then least_squares searches the nonlinear terms only and the
    # linear ones are solved exactly for each of its trials (variable
    # projection) - worth it where a model evaluation is expensive against the
    # linear solve, as in zpnfit.
    #
    # A subclass opts in by implementing linear_term() and residual_vector();
    # linear_columns() is optional and only makes building the columns cheaper.

    def linear_term(self, term):
        """True if the model is linear in this term (for any values of the others)"""
        return False

    def residual_vector(self, values, data):
        """Signed weighted residuals, shape (m, N): m components per object.

        Must be affine in the linear terms; sum over m of squares per object
        is the squared fit_residuals() of the object without delin.
        """
        return None

    def linear_columns(self, values, lin, data):
        """Derivatives of residual_vector() by the linear terms, shape (m*N, len(lin)).

        Default: the model is affine in them, so column k = r(e_k) - r(0).
        """
        v = np.array(values, dtype=float)
        v[lin] = 0.0
        r0 = self.residual_vector(v, data).ravel()
        cols = []
        for i in lin:
            v[i] = 1.0
            cols.append(self.residual_vector(v, data).ravel() - r0)
            v[i] = 0.0
        return np.column_stack(cols) if cols else np.zeros((len(r0), 0))

    def _split_linear(self, bounds):
        """Indices of the linear and of the nonlinear fitted terms; bounded terms count as nonlinear"""
        if self.residual_vector.__func__ is termfit.residual_vector:
            return [], list(range(len(self.fitterms)))
        lin = [i for i, (term, b) in enumerate(zip(self.fitterms, bounds))
               if self.linear_term(term) and not np.isfinite(b).any()]
        non = [i for i in range(len(self.fitterms)) if i not in lin]
        return lin, non

    def _solve_linear(self, values, lin, data, max_irls=50, weights=None):
        """Solve the linear terms with the others fixed at values: with given
        per-object weights a single weighted least-squares solve, otherwise
        with delin the Cauchy loss by IRLS, else plain least squares.

        Returns full value vector, residual_vector at it, and the unscaled columns.
        """
        v = np.array(values, dtype=float)
        v[lin] = 0.0
        r0 = self.residual_vector(v, data)
        m, n = r0.shape
        J = self.linear_columns(v, lin, data)
        scale = np.sqrt(np.sum(J * J, axis=0))
        scale[scale == 0] = 1.0
        Js = J / scale

        irls = weights is None and self.delin
        w = np.ones(n) if weights is None else weights
        prev = np.inf
        for _ in range(max_irls if irls else 1):
            sw = np.tile(np.sqrt(w), m)
            A = Js * sw[:, None]
            b = -r0.ravel() * sw
            try:
                # normal equations: N x K with K << N, columns of unit norm
                c = cho_solve(cho_factor(A.T @ A), A.T @ b)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, b, rcond=None)[0]
            r = (r0.ravel() + Js @ c).reshape(m, n)
            if not irls:
                break
            d2 = np.sum(r * r, axis=0)
            cost = np.sum(np.log1p(d2))
            if prev - cost <= 1e-12 * cost:
                break
            prev = cost
            w = 1.0 / (1.0 + d2)
        v[lin] = c / scale
        return v, r, J

    def _fit_linear(self, data, lin):
        """Fit with all fitted terms linear; returns values and the Jacobian of fit_residuals.

        fit_residuals() is the length d of each object's residual_vector, or
        cauchy_delin(d) with delin, so its Jacobian follows from the columns
        by the chain rule, without further model evaluations.
        """
        values, r, J = self._solve_linear(self.fitvalues, lin, data)
        return values, self._jac_linear(r, J)

    def _jac_linear(self, r, J):
        """Jacobian of fit_residuals by the linear terms from residual_vector r
        and its columns J: fit_residuals is the length d of each object's r, or
        cauchy_delin(d) with delin, so it follows by the chain rule"""
        m, n = r.shape
        d = np.sqrt(np.sum(r * r, axis=0))
        jac = np.einsum('mn,mnk->nk', r, J.reshape(m, n, -1)) / np.where(d > 0, d, 1.0)[:, None]
        if self.delin:
            f = np.sqrt(np.log1p(d * d))
            # d(f)/d(d); f ~ d for small d, so the limit at d = 0 is 1
            fprime = np.where(d > 0, d / ((1 + d * d) * np.where(f > 0, f, 1.0)), 1.0)
            jac = jac * fprime[:, None]
        return jac

    def _fit_varpro(self, data, lin, non, lower, upper):
        """Variable projection: least_squares searches the nonlinear terms only,
        the linear ones are solved exactly for each of its trials.  Returns
        values and the Jacobian of fit_residuals.

        The search minimizes plain least squares, where each trial is a single
        solve.  With Cauchy delin, the linear terms are then solved under the
        Cauchy loss (IRLS) with the nonlinear ones as found: a Cauchy objective
        for the search itself means an IRLS of ~40 rounds per trial, and a
        least_squares polish of all terms afterwards took 3x the time of the
        whole fit for a final WSSR within ~1% (fits of the ~/test frames,
        whose outliers refine_fit has already clipped at 3 sigma)."""
        values = np.array(self.fitvalues, dtype=float)
        plain = np.ones(len(data[0]))   # unit weights: plain least squares, no IRLS

        def outer(theta):
            v = values.copy()
            v[non] = theta
            _, r, _ = self._solve_linear(v, lin, data, weights=plain)
            return r.ravel()

        res = fit.least_squares(outer, values[non], ftol=1e-15, x_scale='jac',
                                bounds=(np.array(lower)[non], np.array(upper)[non]))
        values[non] = res.x
        values, r, J = self._solve_linear(values, lin, data, weights=plain)

        if self.delin:
            # the linear terms under the Cauchy loss, the nonlinear ones as found
            values, r, J = self._solve_linear(values, lin, data)

        jac = np.empty((r.shape[1], len(values)))
        jac[:, lin] = self._jac_linear(r, J)

        def at(theta):
            v = values.copy()
            v[non] = theta
            return self.fit_residuals(v, data)
        jac[:, non] = approx_derivative(at, values[non], method='2-point')
        return values, jac

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
