"""
afterglow - new (uncatalogued) stationary sources in a frame sequence, their light curves
judged for a GRB/EP afterglow: is it there, does it vary, is the variation ordered, and
where is it heading.

The input is what transients.py collects: the per-frame detections of every candidate
(not in the reference catalogue, or brightened) and the per-frame records of the
catalogued field stars (the flare search's), which give each frame's common mode (the
zeropoint residual) and its systematic floor.  For every stationary candidate that is not
a catalogued star (those are the flare search's business):

  - the light curve corrected for the frames' common mode, errors with the frame floor,
  - variation: chi2/ndf of a constant, as a significance Z_VAR,
  - continuity: the length of the light curve, sum |dm|/sigma of successive points, against
    its scatter, normalised so that pure noise gives 1 (1% of noise below 0.64 for 12
    points); an ordered curve (rise, decay, rise+decay) of 10 sigma amplitude gives ~0.35,
    whatever its shape (the "string length" of the phase-dispersion methods),
  - a power law in time since the trigger, m = a + 2.5 alpha log10(t - T0) (alpha > 0
    fading), or a linear trend in mag/h without a trigger time; the curvature of a
    quadratic in log t tells a rise turning into a decay,
  - extrapolated magnitude +5, +15, +60 min after the last point from the power law of the
    recent half in log time (t - T0 >= (t_last - T0)/2; an afterglow is often caught rising or
    flaring); the error adds the disagreement with the whole curve's power law; a rise is held
    beyond the source's age; without a trigger time a straight line, at most one span ahead,
  - error-box distance (trigger from pyrt.utils.grbinfo), persistence (fraction of frames),
    margin above the detection limit.

Class: NEW (steady within the errors), FADING / RISING (significant variation, ordered,
the trend's sign), TURNING (ordered, curved: a rise into a decay or the reverse), ERRATIC
(significant variation, not ordered: a bogus or a flaring source - not rejected, a GRB may
flare).  S ranks them: margin above the limit, error box, persistence, order.
"""

import numpy as np
import astropy.table

ORDERED = 0.6      # normalised continuity below this: ordered (1% of pure noise, 12 points)
Z_VAR = 3.0        # variation significance for "varies"
MIN_FRAC = 0.75    # present in at least this fraction of the frames
MAX_MOTION = 3.0   # motion significance above this, and
MIN_MOVE = 1.0     # [arcsec] a displacement over the span above this: a mover, not ours
                   # (a bright source has centroid errors far below the frames' systematics,
                   # saturation, blends: GRB 210619B at 12 mag "moved" 5.7 sigma = 0.45")
EXTRAP = (5.0, 15.0, 60.0)  # [min] after the last point
ALPHA_FLOOR = 0.5  # the least uncertainty of the decay index in an extrapolation


def frame_stats(rec):
    """{frame: (jd, common mode, floor, limit)} from the flare search's field-star records
    (transients.py flare_rec as a table: FRAME, JD, MAG, ERR, EXPECT, LIM)."""
    from pyrt.core.flares import _floor
    out = {}
    for fr in np.unique(rec['FRAME']):
        m = rec['FRAME'] == fr
        e = np.asarray(rec['EXPECT'][m] - rec['MAG'][m], dtype=float)
        err = np.asarray(rec['ERR'][m], dtype=float)
        fin = np.isfinite(e) & np.isfinite(err)
        if not np.any(fin):
            continue
        cut = max(0.05, float(np.percentile(err[fin], 20)))
        ok = fin & (err < cut)
        cm = float(np.median(e[ok])) if np.sum(ok) >= 3 else 0.0
        s = _floor(e - cm, err, ok)
        lim = float(np.median(rec['LIM'][m])) if 'LIM' in rec.colnames else np.nan
        out[int(fr)] = (float(np.median(rec['JD'][m])), cm, s, lim)
    return out


def continuity(m, s):
    """Normalised light-curve length: 1 for pure noise, -> ~0.25 for a clean monotonic curve."""
    n = len(m)
    if n < 4:
        return np.nan, np.nan
    w = 1 / s**2
    m0 = np.sum(w * m) / np.sum(w)
    chi = np.sqrt(np.sum(((m - m0) / s)**2) / (n - 1))
    length = np.sum(np.abs(np.diff(m)) / np.sqrt(s[:-1]**2 + s[1:]**2)) * n / (n - 1)
    return (length / chi / (0.798 * n) if chi > 0 else np.nan), chi


def _wfit(x, y, s, deg):
    """Weighted polynomial fit, coefficients (highest first) and covariance scaled by
    chi2/ndf when > 1."""
    A = np.vander(x, deg + 1)
    W = 1 / s
    p, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    cov = np.linalg.inv((A * W[:, None]).T @ (A * W[:, None]))
    ndf = max(len(x) - deg - 1, 1)
    chi2 = np.sum(((y - A @ p) * W)**2) / ndf
    return p, cov * max(chi2, 1.0), chi2


def assess(t_jd, m, s, t0_jd=None):
    """One light curve (mid-exposure JD, corrected magnitudes, errors with the floor).
    Returns a dict of the light-curve properties and extrapolations."""
    o = np.argsort(t_jd)
    t_jd, m, s = t_jd[o], m[o], s[o]
    n = len(m)
    w = 1 / s**2
    mean = float(np.sum(w * m) / np.sum(w))
    chi2 = float(np.sum(((m - mean) / s)**2) / max(n - 1, 1))
    # significance of the variation: chi2 with n-1 dof as a normal deviate (Wilson-Hilferty)
    k = max(n - 1, 1)
    z_var = float(((chi2) ** (1 / 3) - (1 - 2 / (9 * k))) / np.sqrt(2 / (9 * k)))
    cont, _ = continuity(m, s)
    r = dict(N=n, MAG_MEAN=mean, MAG_LAST=float(m[-1]), CHI2=chi2, Z_VAR=z_var, CONT=float(cont),
             ALPHA=np.nan, ALPHA_ERR=np.nan, RATE=np.nan, RATE_ERR=np.nan, CURV_Z=np.nan,
             T_LAST=float(t_jd[-1]), SPAN=float((t_jd[-1] - t_jd[0]) * 1440))
    # linear trend in time [mag/h]
    th = (t_jd - t_jd[0]) * 24
    p, cov, _ = _wfit(th, m, s, 1)
    r['RATE'], r['RATE_ERR'] = float(p[0]), float(np.sqrt(cov[0, 0]))
    r['ALPHA_RECENT'] = r['ALPHA_RECENT_ERR'] = np.nan
    span = (t_jd[-1] - t_jd[0]) * 1440
    if t0_jd is not None and np.all(t_jd > t0_jd):
        x = np.log10((t_jd - t0_jd) * 1440)
        p, cov, _ = _wfit(x, m, s, 1)
        r['ALPHA'], r['ALPHA_ERR'] = float(p[0] / 2.5), float(np.sqrt(cov[0, 0]) / 2.5)
        if n >= 5 and np.ptp(x) > 0:
            q, qcov, _ = _wfit(x - x.mean(), m, s, 2)
            r['CURV_Z'] = float(q[0] / np.sqrt(qcov[0, 0])) if qcov[0, 0] > 0 else np.nan
        # the recent half in log time (t - T0 >= half of the last): an afterglow is often
        # caught rising or flaring, and the early points (bright, precise) would set the slope
        rec = (t_jd - t0_jd) >= (t_jd[-1] - t0_jd) / 2
        if np.sum(rec) < 4:
            rec = np.arange(n) >= n - min(n, 4)
        pr, covr, _ = _wfit(x[rec], m[rec], s[rec], 1)
        r['ALPHA_RECENT'], r['ALPHA_RECENT_ERR'] = float(pr[0] / 2.5), float(np.sqrt(covr[0, 0]) / 2.5)
        age = (t_jd[-1] - t0_jd) * 1440
        for dt in EXTRAP:
            # a rise is not extrapolated beyond the source's age (t_last - T0): held there
            xx = np.log10(age + (dt if pr[0] >= 0 else min(dt, age)))
            a_ = np.array([xx, 1.0])
            mr, mall = float(a_ @ pr), float(a_ @ p)
            # the error: the recent fit's own, and the disagreement with the whole curve's
            # (one law for the whole curve is a model, flaring breaks it)
            err = np.hypot(np.sqrt(a_ @ covr @ a_), mr - mall)
            if np.isfinite(r['CURV_Z']) and abs(r['CURV_Z']) >= 3:
                # a curved light curve (a flare's decline, a rise turning over) is about to
                # change its character: a flattening is as likely as the trend going on
                # (GRB 180325A: the steep decline after the flare ran into a plateau)
                err = np.hypot(err, mr - m[-1])
            # and alpha is not known better than ALPHA_FLOOR a few minutes after a trigger:
            # breaks happen (GRB 250702F steepened after 9 min, unforeseeable before)
            err = np.hypot(err, 2.5 * ALPHA_FLOOR * (np.log10(age + dt) - x[-1]))
            r['MAG_%d' % dt] = mr
            r['MAG_%d_ERR' % dt] = float(err)
    else:
        # no trigger time: a straight line, and not further than the span it was fitted on
        p, cov, _ = _wfit(th, m, s, 1)
        for dt in EXTRAP:
            if dt > span:
                r['MAG_%d' % dt] = r['MAG_%d_ERR' % dt] = np.nan
                continue
            a_ = np.array([(t_jd[-1] - t_jd[0]) * 24 + dt / 60, 1.0])
            r['MAG_%d' % dt] = float(a_ @ p)
            r['MAG_%d_ERR' % dt] = float(np.sqrt(a_ @ cov @ a_))
    return r


def afterglow_search(cands, frames, nframes, trigger=None, verbose=False):
    """cands: list of dicts with RA, DEC, MOTION (significance), JD (array, mid-exposure), MAG,
    ERR (arrays), CATMAG (array or nan); frames: frame_stats() output; nframes: frames in the
    sequence; trigger: grbinfo record or None.  Returns an astropy Table, best first, or None."""
    fjd = np.array([v[0] for v in frames.values()]) if frames else np.array([])
    fval = list(frames.values())
    lim_all = np.nanmedian([v[3] for v in fval]) if fval else np.nan
    t0 = trigger['jd0'] if trigger is not None and np.isfinite(trigger.get('jd0', np.nan)) else None
    rows = []
    for c in cands:
        jd_ = np.asarray(c['JD'], dtype=float)
        move = float(c.get('MOVE', np.inf))  # [arcsec/h]
        shift = move * (np.max(jd_) - np.min(jd_)) * 24 if len(jd_) else np.inf
        if c['MOTION'] > MAX_MOTION and not shift < MIN_MOVE:
            continue
        catm = np.asarray(c['CATMAG'], dtype=float)
        if np.sum(np.isfinite(catm)) > len(catm) / 2:
            continue                      # a catalogued star: the flare search's
        if len(c['MAG']) < MIN_FRAC * nframes:
            continue
        jd, m, e = (np.asarray(c[k], dtype=float) for k in ('JD', 'MAG', 'ERR'))
        cm = np.zeros(len(jd))
        fl = np.zeros(len(jd))
        if len(fjd):
            idx = np.argmin(np.abs(jd[:, None] - fjd[None, :]), axis=1)
            cm = np.array([fval[i][1] for i in idx])
            fl = np.array([fval[i][2] for i in idx])
        mc = m + cm                       # E = expect - mag: stars measured c too bright
        s = np.sqrt(np.maximum(e, 0.005)**2 + fl**2)
        r = assess(jd, mc, s, t0)
        r['RA'], r['DEC'] = float(np.mean(c['RA'])), float(np.mean(c['DEC']))
        r['FRAC'] = len(m) / nframes
        r['MOTION'] = float(c['MOTION'])
        r['SHIFT'] = float(shift)
        r['MARGIN'] = float(lim_all - r['MAG_MEAN']) if np.isfinite(lim_all) else np.nan
        if trigger is not None and np.isfinite(trigger.get('ra', np.nan)):
            from pyrt.utils.grbinfo import errorbox_distance
            d, db = errorbox_distance(trigger, r['RA'], r['DEC'])
            r['DIST'], r['DIST_BOX'] = float(d * 60), float(db)
        else:
            r['DIST'] = r['DIST_BOX'] = np.nan
        # class
        varies = r['Z_VAR'] >= Z_VAR
        ordered = np.isfinite(r['CONT']) and r['CONT'] < ORDERED
        if not varies:
            cls = 'NEW'
        elif not ordered:
            cls = 'ERRATIC'
        elif np.isfinite(r['CURV_Z']) and abs(r['CURV_Z']) >= 3:
            cls = 'TURNING'   # a rise into a decay (or back): the extrapolation is not to be trusted
        else:
            slope, err = ((r['ALPHA_RECENT'], r['ALPHA_RECENT_ERR']) if np.isfinite(r['ALPHA_RECENT'])
                          else (r['RATE'], r['RATE_ERR']))
            cls = 'FADING' if slope > 0 else 'RISING'
        r['CLASS'] = cls
        # score: how far above the limit (a faint source near it is often noise), error box,
        # persistence, order
        sc = np.clip(r['MARGIN'], 0, 5) if np.isfinite(r['MARGIN']) else 1.0
        if np.isfinite(r['DIST_BOX']):
            sc += 2.5 if r['DIST_BOX'] <= 1 else (1.0 if r['DIST_BOX'] <= 2 else 0.0)
        sc += 2 * (r['FRAC'] - MIN_FRAC) / (1 - MIN_FRAC)
        sc += 1.5 if cls in ('FADING', 'RISING', 'TURNING') else (-1.0 if cls == 'ERRATIC' else 0.0)
        r['S'] = float(sc)
        rows.append(r)
    if not rows:
        return None
    res = astropy.table.Table(rows=rows)
    res.sort('S', reverse=True)
    if verbose:
        print("afterglow: %d new stationary sources (of %d candidates), %d frames, limit %.2f"
              % (len(res), len(cands), nframes, lim_all))
    return res


def report_lines(res, trigger=None, n=10):
    out = []
    for r in res[:n]:
        trend = ("alpha %+.2f±%.2f (recent %+.2f±%.2f)" % (r['ALPHA'], r['ALPHA_ERR'], r['ALPHA_RECENT'], r['ALPHA_RECENT_ERR'])
                 if np.isfinite(r['ALPHA']) else "%+.2f±%.2f mag/h" % (r['RATE'], r['RATE_ERR']))
        box = (" %.1f' = %.2f box" % (r['DIST'], r['DIST_BOX'])) if np.isfinite(r['DIST']) else ""
        out.append("%-7s %.5f %+.5f mag %.2f (last %.2f, %.1f above limit) var %.1fσ cont %.2f %s "
                   "-> +5' %.2f±%.2f +15' %.2f±%.2f +60' %.2f±%.2f S=%.1f |%s n=%d/%.1f min"
                   % (r['CLASS'], r['RA'], r['DEC'], r['MAG_MEAN'], r['MAG_LAST'], r['MARGIN'], r['Z_VAR'],
                      r['CONT'], trend, r['MAG_5'], r['MAG_5_ERR'], r['MAG_15'], r['MAG_15_ERR'],
                      r['MAG_60'], r['MAG_60_ERR'], r['S'], box, r['N'], r['SPAN']))
    return out
