"""
flares - flaring (catalogued, usually red dwarf) stars in a sequence of dophot ECSVs.

A flare star is a catalogued star that is (a) brighter than its catalogue-derived
quiescent magnitude and (b) fading.  Neither alone is enough: very red stars are
outside the colour range the photometric model is calibrated on (and many of them
are variable), and a slope alone is noise for most stars.  Both are measured
against the field, so that they hold for D50 kinetic sub-frame sums (11 s), D50
mean frames (110 s) and SBT 12 s frames alike:

  excess   e = expected(catalog, colour model) - MAG_CALIB         (>0 brighter)
           per frame the common median over the field is removed and a
           systematic floor s is fitted so that the field's e/sigma has unit
           (robust) scatter; colour extrapolation beyond the field's 5-95%
           colour range adds red_margin [mag per mag of colour] in quadrature.
  trend    e(t) = a + b*(t - t_first), weighted fit per star; Z_fade = -b/sigma_b,
           sigma_b scaled by the residual chi2 of a quadratic (so the curvature
           of a decaying flare does not count as noise) and by the robust width
           of Z_fade over all the field stars.
  Z_ex     = a/sqrt((k*sigma_a)^2 + colour margin^2), k the field normalisation of a/sigma_a,
           sigma_a includes the frame floor (catalogue/model error, does not average down).

  S = (Z_ex+ + Z_fade+)/sqrt(2)   (Stouffer, only the flare-like signs count; one frame: S = Z_ex)

  Both normalisations use the field stars within the same magnitude range (the
  noise of faint stars has heavier tails).  Only stars expected well above the
  frame limit (depth mag brighter than MAGZERO+LIMFLX3, median over frames; red stars
  only depth_prior) and found in at least half of the frames are used,
  otherwise the faint end is all Eddington bias.

Priors P (added to S): in the trigger error box +1, red (colour beyond the calibrated
range) +0.5, and if red also a dwarf by reduced proper motion H = m + 5 log(pm["/yr]) + 5
> 13 +1 (M giants of the same colour have H ~ 10; bluer dwarfs are not EP flarers).

Classes: FLARE  Z_ex>=1.5 and Z_fade>=2 and S+P >= flare_sig
         FLARE? Z_ex>=1.5 and not brightening (Z_fade > -2) and S+P >= 3,
                and the star red, or S >= 5 and Z_fade >= 2 on its own (the error box alone holds
                tens of stars, and many EP triggers are GRBs with no flare star at all)
(a constant excess without fading is not a flare: transients.py lists it as BRIGHT).
Both: stars admitted only by depth_prior need Z_ex >= 3 and S >= 3; the last point must not
be more than 2 sigma below the catalogue (a flare decays to quiescence, not below).
"""

import numpy as np
import astropy.table
from sklearn.neighbors import KDTree

from pyrt.utils.grbinfo import errorbox_distance

DWARF_H = 13.0       # reduced proper motion above which a star is (most likely) a dwarf
RED_X = 0.3          # colour extrapolation [mag] above which a star counts as "red"
MIN_FIELD = 10       # stars needed for the field normalisation
NEIGHBOURS = 40      # stars of similar magnitude for the local normalisation
PRIOR_INBOX, PRIOR_DWARF, PRIOR_RED = 1.0, 1.0, 0.5
NOPRIOR_S = 5.0      # S needed for a candidate that is neither red nor a dwarf


def _local_scale(z, mag, normal):
    """Robust centre and width (>= 1) of z among the NEIGHBOURS 'normal' stars closest in
    magnitude (the star itself excluded): a magnitude-dependent drift (e.g. all faint stars
    fading with changing seeing or sky) shifts the centre, not the width."""
    c, k = np.zeros(len(z)), np.ones(len(z))
    ref = np.where(normal & np.isfinite(z))[0]
    if len(ref) < MIN_FIELD:
        return c, k
    nn = min(NEIGHBOURS, len(ref))
    for j in range(len(z)):
        near = ref[np.argsort(np.abs(mag[ref] - mag[j]))[:nn+1]]
        near = near[near != j][:nn]
        c[j] = np.median(z[near])
        w = _mad_sigma(z[near])
        k[j] = max(w, 1.0) if np.isfinite(w) else 1.0
    return c, k


def _mad_sigma(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.4826*np.median(np.abs(x - np.median(x)))


def _floor(e, err, ok):
    """Systematic floor s such that robust std(e/sqrt(err^2+s^2)) == 1 (0 if already <= 1)."""
    e, err = e[ok], err[ok]
    if len(e) < MIN_FIELD:
        return 0.0
    def width(s):
        return _mad_sigma(e/np.sqrt(err**2 + s**2))
    if width(0.0) <= 1.0:
        return 0.0
    lo, hi = 0.0, max(0.5, 3*_mad_sigma(e))
    for _ in range(40):
        mid = 0.5*(lo + hi)
        if width(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def _star_ids(ra, dec, radius_arcsec=1.0):
    """Same catalogue star in different frames (catalogue coordinates, so ~identical)."""
    P = np.array([ra*np.cos(np.radians(dec)), dec]).T*3600.0
    tree = KDTree(P)
    ids = np.full(len(ra), -1, dtype=int)
    nid = 0
    for i, nb in enumerate(tree.query_radius(P, r=radius_arcsec)):
        if ids[i] >= 0:
            continue
        free = nb[ids[nb] < 0]
        ids[free] = nid
        nid += 1
    return ids


def _linfit(t, y, s):
    """Weighted y = a + b t. Returns a, b, sa, sb, chi2_quad/ndf (residual of a quadratic)."""
    w = 1.0/s**2
    A = np.vstack([np.ones_like(t), t]).T
    cov = np.linalg.inv(A.T @ (A*w[:, None]))
    a, b = cov @ A.T @ (w*y)
    sa, sb = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    rchi = 1.0
    if len(t) >= 5:
        Q = np.vstack([np.ones_like(t), t, t**2]).T
        p = np.linalg.lstsq(Q*np.sqrt(w)[:, None], y*np.sqrt(w), rcond=None)[0]
        rchi = np.sum(w*(y - Q @ p)**2)/(len(t) - 3)
    return a, b, sa, sb, rchi


def flare_search(rec, trigger=None, red_margin=0.05, flare_sig=5.0, window=None, depth=1.5, depth_prior=0.7,
                 verbose=False):
    """rec: table of catalogue-matched stars from all frames, columns
         FRAME JD CATRA CATDEC MAG ERR EXPECT CATMAG C0..C3 PM X Y
       (ERR already scaled by sqrt(WSSRNDF), EXPECT the model magnitude from the catalogue).
       trigger: grbinfo record or None.  window [h]: use only data this long after
       the first frame (None = all).  depth [mag]: use stars expected this much above
       the frame limit LIM (column LIM, optional), depth_prior for red stars.
       Returns an astropy Table, one row per star, sorted most interesting first."""
    rec = astropy.table.Table(rec, copy=True)
    nframes = len(np.unique(rec['FRAME'])) if len(rec) else 0
    if len(rec) == 0:
        return None
    rec['STAR'] = _star_ids(np.asarray(rec['CATRA']), np.asarray(rec['CATDEC']))
    tstart = np.min(rec['JD'])
    rec['T'] = (rec['JD'] - tstart)*24.0
    rec['E'] = np.asarray(rec['EXPECT'] - rec['MAG'], dtype=np.float64)
    rec['SIG'] = np.nan
    rec['COLX'] = 0.0   # colour extrapolation beyond the field's range, either side (-> margin)
    rec['COLR'] = 0.0   # the same on the red side only (-> "red" prior)
    floors = {}
    colours = ['C0', 'C1', 'C2', 'C3']

    # --- per frame: common mode, systematic floor, colour range ---
    for fr in np.unique(rec['FRAME']):
        m = rec['FRAME'] == fr
        e = np.array(rec['E'][m]); err = np.array(rec['ERR'][m], dtype=np.float64)
        ok = (err < 0.05) & np.isfinite(e)
        if np.sum(ok) >= 3:
            e = e - np.median(e[ok])
        s = _floor(e, err, ok)
        floors[fr] = s
        x2 = np.zeros(np.sum(m))
        r2 = np.zeros(np.sum(m))
        for c in colours:
            cv = np.array(rec[c][m], dtype=np.float64)
            fin = ok & np.isfinite(cv)
            if np.sum(fin) < MIN_FIELD:
                continue
            lo, hi = np.percentile(cv[fin], [5, 95])
            out = np.where(np.isfinite(cv), np.maximum(0, np.maximum(cv - hi, lo - cv)), 0)
            x2 += out**2
            r2 += np.where(np.isfinite(cv), np.maximum(0, cv - hi), 0)**2
        rec['E'][m] = e
        rec['SIG'][m] = np.sqrt(err**2 + s**2)
        rec['COLX'][m] = np.sqrt(x2)
        rec['COLR'][m] = np.sqrt(r2)

    # --- which stars: well above the limit (median over frames, so that a star does not drop
    # in and out with the conditions); red stars are allowed closer to it (depth_prior),
    # the Eddington bias only matters for the crowd; found in at least half of the frames ---
    keep = np.zeros(len(rec), dtype=bool)
    for sid in np.unique(rec['STAR']):
        m = rec['STAR'] == sid
        if np.sum(m) < 0.5*nframes:
            continue
        if 'LIM' in rec.colnames:
            likely = np.median(rec['COLR'][m]) > RED_X
            # a red star counts with its measured brightness if brighter (a flaring star
            # well detected although quiescent near the limit); it still has to pass the
            # stricter near-limit test below
            # (quiescent at least near detectable: a catalogue star far below the limit "seen"
            # several mag brighter is a misidentification, not a flare)
            expm, limm = np.median(rec['EXPECT'][m]), np.median(rec['LIM'][m])
            ref = min(expm, np.median(rec['MAG'][m])) if likely and expm < limm else expm
            if ref >= np.median(rec['LIM'][m]) - (depth_prior if likely else depth):
                continue
        keep |= m
    rec = rec[keep]
    if len(rec) == 0:
        return None

    # --- per star ---
    rows = []
    for sid in np.unique(rec['STAR']):
        o = rec[rec['STAR'] == sid]
        o = o[np.argsort(o['T'])]
        if window is not None:
            o = o[o['T'] <= o['T'][0] + window]
        t = np.array(o['T'] - o['T'][0], dtype=np.float64)
        e = np.array(o['E'], dtype=np.float64)
        s = np.array(o['SIG'], dtype=np.float64)
        n = len(o)
        sfloor = np.median([floors[f] for f in o['FRAME']])
        if n >= 3 and np.ptp(t) > 0:
            a, b, sa, sb, rchi = _linfit(t, e, s)
            scale = np.sqrt(max(rchi, 1.0))
            sa *= scale; sb *= scale
        else:
            w = 1/s**2
            a = np.sum(w*e)/np.sum(w); sa = 1/np.sqrt(np.sum(w)); b = sb = np.nan
            if n == 2 and np.ptp(t) > 0:
                b = (e[1] - e[0])/(t[1] - t[0]); sb = np.hypot(s[0], s[1])/(t[1] - t[0])
        colx = float(np.median(o['COLX']))
        colr = float(np.median(o['COLR']))
        scol = red_margin*colx
        catmag = float(np.median(o['CATMAG']))
        pm = float(np.median(o['PM']))
        H = catmag + 5*np.log10(pm/1000.0) + 5 if pm > 0 else np.nan
        dist, rbox = errorbox_distance(trigger, float(np.median(o['CATRA'])), float(np.median(o['CATDEC'])))
        # admitted only thanks to depth_prior (red, within `depth` of the limit): noisier
        nearlim = bool('LIM' in o.colnames and np.median(o['EXPECT']) >= np.median(o['LIM']) - depth)
        rows.append((sid, n, float(np.median(o['CATRA'])), float(np.median(o['CATDEC'])),
                     float(np.median(o['MAG'])), catmag, a, np.sqrt(sa**2 + sfloor**2), scol,
                     b, sb, float(np.max(e)), float(e[-1]), float(np.hypot(s[-1], scol)), nearlim, colx, colr,
                     float(np.median(o['C0'])), float(np.median(o['C1'])), float(np.median(o['C2'])),
                     pm, H, dist*60 if np.isfinite(dist) else np.nan, rbox, float(np.ptp(o['T'])*60)))
    res = astropy.table.Table(rows=rows, names=['STAR', 'NUM', 'ALPHA_J2000', 'DELTA_J2000', 'MAG_CALIB', 'CATMAG',
        'EXCESS', 'EXCESS_ERR', 'COLOR_ERR', 'TREND', 'TREND_ERR', 'EXCESS_MAX', 'EXCESS_LAST', 'LAST_ERR', 'NEARLIM', 'COLX', 'COLR',
        'C0', 'C1', 'C2', 'PM', 'HRPM', 'DIST', 'DIST_BOX', 'SPAN'])

    # --- field normalisation (the field's stars are mostly constant) ---
    # the floor is mostly catalogue/model error, constant for a star: it does not average down with
    # more frames; the colour margin is a systematic of its own, kept out of the field normalisation
    zex = res['EXCESS']/res['EXCESS_ERR']
    zf = -res['TREND']/res['TREND_ERR']
    normal = np.abs(zex) < 3
    mag = np.array(res['MAG_CALIB'], dtype=np.float64)
    # magnitude-dependent systematics: the neighbours' median trend [mag/h] and excess [mag]
    # are subtracted (e.g. all faint stars drifting with seeing/sky), then the widths normalised
    c_t, _ = _local_scale(np.array(res['TREND'], dtype=np.float64), mag, normal)
    c_e, _ = _local_scale(np.array(res['EXCESS'], dtype=np.float64), mag, normal)
    res['TREND'] -= c_t
    for c in ('EXCESS', 'EXCESS_MAX', 'EXCESS_LAST'):
        res[c] -= c_e
    zex = res['EXCESS']/res['EXCESS_ERR']
    zf = -res['TREND']/res['TREND_ERR']
    _, k_ex = _local_scale(np.array(zex), mag, normal)
    _, k_f = _local_scale(np.array(zf), mag, normal)
    res['EXCESS_ERR'] = np.sqrt((k_ex*res['EXCESS_ERR'])**2 + res['COLOR_ERR']**2)
    res['Z_EX'] = res['EXCESS']/res['EXCESS_ERR']
    res['Z_FADE'] = zf/k_f
    res.meta['K_EX'] = float(np.median(k_ex))
    res.meta['K_FADE'] = float(np.median(k_f))
    res.meta['FLOOR'] = float(np.median(list(floors.values()))) if floors else 0.0
    if verbose:
        print("flares: %d stars, %d frames, field normalisation excess x%.2f, trend x%.2f, median floor %.3f mag"
              % (len(res), len(floors), res.meta['K_EX'], res.meta['K_FADE'], res.meta['FLOOR']))

    zp_ex = np.maximum(np.nan_to_num(res['Z_EX']), 0)
    zp_f = np.maximum(np.nan_to_num(res['Z_FADE']), 0)
    has_trend = np.isfinite(res['Z_FADE'])
    res['S'] = np.where(has_trend, (zp_ex + zp_f)/np.sqrt(2), zp_ex)
    res['RED'] = res['COLR'] > RED_X  # odd blue/inconsistent catalogue colours only widen the margin
    res['DWARF'] = res['HRPM'] > DWARF_H
    inbox = res['DIST_BOX'] <= 1.0
    res['INBOX'] = np.where(np.isfinite(res['DIST_BOX']), inbox, False)
    # a dwarf by proper motion is only a flare-star prior when red (G/K dwarfs are not EP flarers)
    res['PRIOR'] = PRIOR_INBOX*res['INBOX'] + PRIOR_DWARF*(res['DWARF'] & res['RED']) + PRIOR_RED*res['RED']
    sp = res['S'] + res['PRIOR']
    zf0 = np.nan_to_num(res['Z_FADE'])
    cls = np.full(len(res), '', dtype='U6')
    # without the red prior it has to fade: a constant excess is a catalogue/model problem (BRIGHT)
    stellar = res['RED'] | ((res['S'] >= NOPRIOR_S) & (zf0 >= 2))
    # near the limit (admitted by depth_prior only): the excess itself has to be clear
    solid = ~res['NEARLIM'] | ((res['Z_EX'] >= 3) & (res['S'] >= 3))
    # a flare decays back to quiescence, not below it (that is a variable / eclipsing star):
    # neither the last point nor the fitted line at the end may be significantly below zero
    span = np.array(res['SPAN'], dtype=np.float64)/60.0
    tr = np.nan_to_num(np.array(res['TREND'], dtype=np.float64))
    end = res['EXCESS'] + tr*span
    end_err = np.sqrt(res['EXCESS_ERR']**2 + (np.nan_to_num(res['TREND_ERR'])*span)**2)
    res['EXCESS_END'] = end
    sane = ~(res['EXCESS_LAST'] < -2*res['LAST_ERR']) & ~(end < -2*end_err)
    base = (res['Z_EX'] >= 1.5) & solid & sane
    cls[base & (zf0 > -2) & (sp >= 3) & stellar] = 'FLARE?'
    cls[base & (zf0 >= 2) & (sp >= flare_sig) & stellar] = 'FLARE'
    res['CLASS'] = cls
    # most interesting first: class, inside the error box, then significance with the priors
    key = [-(cls == 'FLARE').astype(int), -(cls != '').astype(int), -res['INBOX'].astype(int), -sp]
    res = res[np.lexsort(key[::-1])]
    return res


def format_row(r, color_names=('C0', 'C1', 'C2')):
    """One-line human readable summary of a flare_search() row."""
    trend = ("%+.2f±%.2f mag/h (%+.1fσ)" % (-r['TREND'], r['TREND_ERR'], r['Z_FADE'])
             if np.isfinite(r['TREND']) else "no trend (1 frame)")
    box = ("%.1f' = %.2f box" % (r['DIST'], r['DIST_BOX'])) if np.isfinite(r['DIST_BOX']) else "no trigger"
    cols = " ".join("%s=%.2f" % (n, r[c]) for n, c in zip(color_names, ('C0', 'C1', 'C2')) if np.isfinite(r[c]))
    flags = ",".join(f for f, v in (('red', r['RED']), ('dwarf', r['DWARF']), ('inbox', r['INBOX'])) if v)
    return ("%-6s %.5f %+.5f mag %.2f excess %+.3f±%.3f (%.1fσ, max %+.3f, now %+.3f) fading %s S=%.1f+%.1f | "
            "%s colx %.2f pm %.0f mas/y H=%.1f | %s [%s] n=%d/%.1f min"
            % (r['CLASS'] or '-', r['ALPHA_J2000'], r['DELTA_J2000'], r['MAG_CALIB'], r['EXCESS'], r['EXCESS_ERR'],
               r['Z_EX'], r['EXCESS_MAX'], r['EXCESS_LAST'], trend, r['S'], r['PRIOR'], cols, r['COLX'], r['PM'], r['HRPM'],
               box, flags, r['NUM'], r['SPAN']))
