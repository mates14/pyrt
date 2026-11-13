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
    beyond the source's age and brightens at most RISE_CAP; a steady source (no significant
    variation) is extrapolated as its mean; without a trigger time a straight line, at most one
    span ahead.  Both fits are kept (about X_REF = log10 of the last point's minutes after T0:
    slope 2.5 alpha, magnitude there, covariance in alpha and mag) and the extrapolation error's
    parts (MAG_n_ESTAT, _EDIFF, _ECURV, _ECAP, _EFLOOR, added in quadrature),
  - frames with the limit CLOUD_DROP above the median are left out,
  - a source whose bins' mean flux is DEEP_EXCESS brighter than the deep image (their sum) shows
    is dropped: not a source but the background (or the photometry) differing between the two,
  - error-box distance (trigger from pyrt.utils.grbinfo), persistence (fraction of frames),
    margin above the detection limit.

Class: SPIKE (all of the variation in one interior bin, kept only within the error box),
NEW (steady within the errors), FADING / RISING (significant variation, ordered,
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
RISE_CAP = 1.0     # [mag] an extrapolated rise brightens at most this much beyond the fit now
CLOUD_DROP = 1.0   # [mag] frames this much shallower than the sequence's median limit are left out
DEEP_EXCESS = 1.0  # [mag] the bins' mean flux this much above the deep image's (their sum): not a source
FAR_BOX = 3.0      # [box radii] a steady NEW source further out is ranked down (catalogue gaps)


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
    for k in ('X_REF', 'WHOLE_M', 'WHOLE_CAA', 'WHOLE_CAM', 'WHOLE_CMM',
              'RECENT_M', 'RECENT_CAA', 'RECENT_CAM', 'RECENT_CMM'):
        r[k] = np.nan
    r['FIT_USED'] = ''
    steady = not r['Z_VAR'] >= Z_VAR
    if t0_jd is not None and np.all(t_jd > t0_jd):
        x = np.log10((t_jd - t0_jd) * 1440)
        xl = x[-1]
        # both fits about the last point: p = (2.5 alpha, magnitude at the last point's time)
        p, cov, _ = _wfit(x - xl, m, s, 1)
        r['ALPHA'], r['ALPHA_ERR'] = float(p[0] / 2.5), float(np.sqrt(cov[0, 0]) / 2.5)
        if n >= 5 and np.ptp(x) > 0:
            q, qcov, _ = _wfit(x - x.mean(), m, s, 2)
            r['CURV_Z'] = float(q[0] / np.sqrt(qcov[0, 0])) if qcov[0, 0] > 0 else np.nan
        # the recent half in log time (t - T0 >= half of the last): an afterglow is often
        # caught rising or flaring, and the early points (bright, precise) would set the slope
        rec = (t_jd - t0_jd) >= (t_jd[-1] - t0_jd) / 2
        if np.sum(rec) < 4:
            rec = np.arange(n) >= n - min(n, 4)
        pr, covr, _ = _wfit(x[rec] - xl, m[rec], s[rec], 1)
        r['ALPHA_RECENT'], r['ALPHA_RECENT_ERR'] = float(pr[0] / 2.5), float(np.sqrt(covr[0, 0]) / 2.5)
        r['X_REF'] = float(xl)
        for k, (pp, cc) in (('WHOLE', (p, cov)), ('RECENT', (pr, covr))):
            r[k + '_M'], r[k + '_CAA'] = float(pp[1]), float(cc[0, 0] / 6.25)
            r[k + '_CAM'], r[k + '_CMM'] = float(cc[0, 1] / 2.5), float(cc[1, 1])
        r['FIT_USED'] = 'recent'
        if abs(pr[0]) < 2 * np.sqrt(covr[0, 0]):
            # the recent half says nothing (a faint source's noisy tail): the whole curve's law
            # (GRB 210312B at ~20.5 mag: the tail's -0.4+-0.8 had made a clear fade "RISING")
            pr, covr = p, cov
            r['FIT_USED'] = 'whole'
        if steady:
            r['FIT_USED'] = 'mean'
        age = (t_jd[-1] - t0_jd) * 1440
        curved = np.isfinite(r['CURV_Z']) and abs(r['CURV_Z']) >= 3
        for dt in EXTRAP:
            e_diff = e_curv = e_cap = 0.0
            if steady:
                # steady within the errors (NEW): no power law, the mean goes on
                mr, e_stat = mean, float(np.sqrt(1 / np.sum(w)))
            else:
                # a rise is not extrapolated beyond the source's age (t_last - T0): held there
                rising = pr[0] < 0
                a_ = np.array([np.log10(age + (min(dt, age) if rising else dt)) - xl, 1.0])
                mr, mall = float(a_ @ pr), float(a_ @ p)
                # the recent fit's own error, and its disagreement with the whole curve's
                # (one law for the whole curve is a model, flaring breaks it)
                e_stat, e_diff = float(np.sqrt(a_ @ covr @ a_)), abs(mr - mall)
                if curved:
                    # a curved light curve (a flare's decline, a rise turning over) is about to
                    # change its character: a flattening is as likely as the trend going on
                    # (GRB 180325A: the steep decline after the flare ran into a plateau)
                    e_curv = abs(mr - m[-1])
                if rising and pr[1] - mr > RISE_CAP:
                    # nor brighter than RISE_CAP above the fit now (EP 260213.832 at D50:
                    # alpha -5.6 of a cloud-driven "rise" held for a doubling was 4.2 mag)
                    e_cap = pr[1] - mr - RISE_CAP
                    mr = float(pr[1] - RISE_CAP)
            # and alpha is not known better than ALPHA_FLOOR a few minutes after a trigger:
            # breaks happen (GRB 250702F steepened after 9 min, unforeseeable before)
            e_floor = 2.5 * ALPHA_FLOOR * (np.log10(age + dt) - xl)
            r['MAG_%d' % dt] = float(mr)
            r['MAG_%d_ERR' % dt] = float(np.sqrt(e_stat**2 + e_diff**2 + e_curv**2 + e_cap**2 + e_floor**2))
            for k, v in (('STAT', e_stat), ('DIFF', e_diff), ('CURV', e_curv), ('CAP', e_cap), ('FLOOR', e_floor)):
                r['MAG_%d_E%s' % (dt, k)] = float(v)
    else:
        # no trigger time: a straight line, and not further than the span it was fitted on
        p, cov, _ = _wfit(th, m, s, 1)
        for dt in EXTRAP:
            for k in ('STAT', 'DIFF', 'CURV', 'CAP', 'FLOOR'):
                r['MAG_%d_E%s' % (dt, k)] = np.nan
            if dt > span:
                r['MAG_%d' % dt] = r['MAG_%d_ERR' % dt] = np.nan
                continue
            a_ = np.array([(t_jd[-1] - t_jd[0]) * 24 + dt / 60, 1.0])
            r['MAG_%d' % dt] = float(a_ @ p)
            r['MAG_%d_ERR' % dt] = r['MAG_%d_ESTAT' % dt] = float(np.sqrt(a_ @ cov @ a_))
    return r


def passer_by(m, s, z_var):
    """True when all of the variation is a single interior bin brighter than both its neighbours
    (by > 3 sigma of its own and > 0.5 mag) and the rest is flat: an asteroid or satellite crossing the (forced) aperture, not a light curve.
    The first and last bins are exempt (an afterglow's earliest point is often its brightest)."""
    if len(m) < 5 or not z_var >= Z_VAR:
        return False
    dev = (np.median(m) - m) / s
    k = int(np.argmax(dev))
    if k == 0 or k == len(m) - 1:
        return False
    step = max(3 * s[k], 0.5)            # the neighbours' own errors may be large: the spike's counts
    if not (m[k - 1] - m[k] > step and m[k + 1] - m[k] > step):
        return False
    keep = np.arange(len(m)) != k
    mk, sk = m[keep], s[keep]
    w = 1 / sk**2
    k_ = len(mk) - 1
    chi2 = np.sum(w * (mk - np.sum(w * mk) / np.sum(w))**2) / k_
    return (chi2 ** (1 / 3) - (1 - 2 / (9 * k_))) / np.sqrt(2 / (9 * k_)) < Z_VAR


def afterglow_search(cands, frames, nframes, trigger=None, verbose=False):
    """cands: list of dicts with RA, DEC, MOTION (significance), JD (array, mid-exposure), MAG,
    ERR (arrays), CATMAG (array or nan); frames: frame_stats() output; nframes: frames in the
    sequence; trigger: grbinfo record or None.  Returns an astropy Table, best first, or None."""
    fjd = np.array([v[0] for v in frames.values()]) if frames else np.array([])
    fval = list(frames.values())
    lim_all = np.nanmedian([v[3] for v in fval]) if fval else np.nan
    t0 = trigger['jd0'] if trigger is not None and np.isfinite(trigger.get('jd0', np.nan)) else None
    # frames much shallower than the rest (clouds, twilight): their calibration and floor come from
    # a few stars, and a sky error of a faint source is scaled up with the lost transparency
    # (EP 260213.832 at D50, airmass 3: zeropoint -1.9 mag over the last bins, three faint NEW
    # objects "rose" 2-3 mag with it)
    cloudy = np.array([np.isfinite(v[3]) and v[3] < lim_all - CLOUD_DROP for v in fval], dtype=bool)
    if verbose and np.any(cloudy):
        print("afterglow: %d of %d frames left out, limit more than %.1f below the median %.2f: %s"
              % (np.sum(cloudy), len(fval), CLOUD_DROP, lim_all, " ".join("%.2f" % fval[i][3] for i in np.flatnonzero(cloudy))))
    nframes_ok = max(nframes - int(np.sum(cloudy)), 1)
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
        jd, m, e = (np.asarray(c[k], dtype=float) for k in ('JD', 'MAG', 'ERR'))
        cm = np.zeros(len(jd))
        fl = np.zeros(len(jd))
        if len(fjd):
            idx = np.argmin(np.abs(jd[:, None] - fjd[None, :]), axis=1)
            ok = ~cloudy[idx]
            jd, m, e, idx = jd[ok], m[ok], e[ok], idx[ok]
            cm = np.array([fval[i][1] for i in idx])
            fl = np.array([fval[i][2] for i in idx])
        if len(m) < MIN_FRAC * nframes_ok or len(m) < 2:
            continue
        mc = m + cm                       # E = expect - mag: stars measured c too bright
        # the deep image is the sum of the bins: a real source, steady or not, has the bins' mean
        # flux there; a background structure is not measured the same in both (EP 260213.832 at
        # D50: a sky ridge at the frame edge, 1.5% of the sky, 17.2 mag in every bin, 19.3 deep)
        dm = float(c.get('DEEPMAG', np.nan))
        fm = float(-2.5 * np.log10(np.mean(10**(-0.4 * mc))))
        if np.isfinite(dm) and fm < dm - DEEP_EXCESS:
            if verbose:
                print("afterglow: %.5f %+.5f dropped: the bins' mean %.2f is %.1f mag brighter than the deep image's %.2f"
                      % (float(np.mean(c['RA'])), float(np.mean(c['DEC'])), fm, dm - fm, dm))
            continue
        s = np.sqrt(np.maximum(e, 0.005)**2 + fl**2)
        r = assess(jd, mc, s, t0)
        o_ = np.argsort(jd)
        spike = passer_by(mc[o_], s[o_], r['Z_VAR'])
        r['RA'], r['DEC'] = float(np.mean(c['RA'])), float(np.mean(c['DEC']))
        r['FRAC'] = len(m) / nframes_ok
        r['N_CLOUDY'] = int(np.sum(cloudy))
        r['MOTION'] = float(c['MOTION'])
        r['SHIFT'] = float(shift)
        r['MARGIN'] = float(lim_all - r['MAG_MEAN']) if np.isfinite(lim_all) else np.nan
        if trigger is not None and np.isfinite(trigger.get('ra', np.nan)):
            from pyrt.utils.grbinfo import errorbox_distance
            d, db = errorbox_distance(trigger, r['RA'], r['DEC'])
            r['DIST'], r['DIST_BOX'] = float(d * 60), float(db)
        else:
            r['DIST'] = r['DIST_BOX'] = np.nan
        # all of the variation in one interior bin: an asteroid or satellite crossing the (forced)
        # aperture looks the same as a short early flare of an afterglow (210312B at T0+75 s);
        # outside the error box it is taken for the crossing, inside it stays as SPIKE
        if spike and not (np.isfinite(r['DIST_BOX']) and r['DIST_BOX'] <= 1) and np.isfinite(r['DIST_BOX']):
            if verbose:
                print("afterglow: %.5f %+.5f dropped: its variation is one bin (a moving object crossing it?), "
                      "%.1f box radii out" % (r['RA'], r['DEC'], r['DIST_BOX']))
            continue
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
            if np.isfinite(r['ALPHA_RECENT']) and abs(r['ALPHA_RECENT']) >= 2 * r['ALPHA_RECENT_ERR']:
                slope = r['ALPHA_RECENT']
            elif np.isfinite(r['ALPHA']):
                slope = r['ALPHA']
            else:
                slope = r['RATE']
            cls = 'FADING' if slope > 0 else 'RISING'
        if spike:
            cls = 'SPIKE'
        r['CLASS'] = cls
        if c.get('FAINT') and not varies:
            continue                      # deep-only, below the catalogue's depth, steady: the field
        # score: how far above the limit (a faint source near it is often noise), error box,
        # persistence, order
        sc = np.clip(r['MARGIN'], 0, 5) if np.isfinite(r['MARGIN']) else 1.0
        if np.isfinite(r['DIST_BOX']):
            sc += 2.5 if r['DIST_BOX'] <= 1 else (1.0 if r['DIST_BOX'] <= 2 else 0.0)
        sc += 2 * (r['FRAC'] - MIN_FRAC) / (1 - MIN_FRAC)
        sc += 1.5 if cls in ('FADING', 'RISING', 'TURNING') else (-1.0 if cls in ('ERRATIC', 'SPIKE') else 0.0)
        if cls == 'NEW' and np.isfinite(r['DIST_BOX']) and r['DIST_BOX'] > FAR_BOX:
            sc -= 3.0     # steady and far from the trigger: a catalogue gap or a saturated star (230818A at SBT)
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


NEAR_LIST = 3.0   # [box radii] candidates this close to the trigger are listed first


def report_lines(res, trigger=None, n=10):
    """The n best candidates as text; those within NEAR_LIST box radii of the trigger come first
    (230818A at SBT: the afterglow, S=4.4 at 2 box radii, was 11th behind steady catalogue gaps)."""
    out = []
    near = [i for i in range(len(res)) if np.isfinite(res['DIST_BOX'][i]) and res['DIST_BOX'][i] <= NEAR_LIST]
    order = near + [i for i in range(len(res)) if i not in near]
    for r in (res[i] for i in order[:n]):
        trend = ("alpha %+.2f±%.2f (recent %+.2f±%.2f)" % (r['ALPHA'], r['ALPHA_ERR'], r['ALPHA_RECENT'], r['ALPHA_RECENT_ERR'])
                 if np.isfinite(r['ALPHA']) else "%+.2f±%.2f mag/h" % (r['RATE'], r['RATE_ERR']))
        box = (" %.1f' = %.2f box" % (r['DIST'], r['DIST_BOX'])) if np.isfinite(r['DIST']) else ""
        out.append("%-7s %.5f %+.5f mag %.2f (last %.2f, %.1f above limit) var %.1fσ cont %.2f %s "
                   "-> +5' %.2f±%.2f +15' %.2f±%.2f +60' %.2f±%.2f S=%.1f |%s n=%d/%.1f min"
                   % (r['CLASS'], r['RA'], r['DEC'], r['MAG_MEAN'], r['MAG_LAST'], r['MARGIN'], r['Z_VAR'],
                      r['CONT'], trend, r['MAG_5'], r['MAG_5_ERR'], r['MAG_15'], r['MAG_15_ERR'],
                      r['MAG_60'], r['MAG_60_ERR'], r['S'], box, r['N'], r['SPAN']))
    return out
