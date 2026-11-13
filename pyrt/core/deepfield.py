"""Transient search in one deep image (a coadd of a late follow-up), no light curve.

Hours after a trigger an afterglow barely changes over the sequence and only the coadd
reaches it; the light-curve searches (afterglow.py, flares.py) then have nothing to work
with.  Here every source of the deep catalogue is matched to what is known at its position
and the rest is checked in the frames the coadd was made of:

  - known: Pan-STARRS DR2 (>= MIN_DET detections; north of -30 deg) and Gaia DR3, one cone
    query per field, cached next to the catalogue; the reference catalogue of the
    calibration (ATLAS, ~19-20 mag) does not reach a coadd's depth (GRB 261113.604 at D50:
    96 of 539 objects of a 37 x 120 s stack "uncatalogued", 79 of them steady field objects)
  - extended: FWHM well above the stars' (a galaxy, not a point source)
  - the frames: forced aperture photometry of each candidate in every input frame (its own
    WCS, the frames scaled to each other by the reference stars); a coadd source that
    comes from one frame (a satellite glint, an aircraft, a cosmic ray) has most of its
    flux there and nothing without it (the same field: 7 such, up to S/N 10 in the coadd)

Classes, best first: NEW (in the frames throughout, consistent with constant), VARIABLE
(in the frames, but not constant), ONEFRAME (artefact of one frame), FAINT (below S/N 3
in the frames together: the coadd's noise or the detection's), EXTENDED; candidates within
the error box first, then by distance in box radii and S/N."""

import os
import warnings

import numpy as np
import astropy.table

MIN_DET = 3          # Pan-STARRS objects with fewer detections are mostly single-epoch junk
MATCH = 2.0          # [arcsec] a deep source this close to a known object is that object
EXT_FWHM = 1.8       # FWHM above this times the stars' median: extended
SN_FRAMES = 3.0      # S/N of the frames together (and without the best one) to be "in the frames"
CHI2_CONST = 3.0     # chi2/ndf of the frame fluxes above this: VARIABLE
REF_MAG = (13.0, 17.5)


def _sep(ra1, de1, ra2, de2):
    """[arcsec], small angles"""
    return np.hypot((ra1 - ra2) * np.cos(np.radians(de1)), de1 - de2) * 3600


def known_sources(ra, dec, radius, cache=None, verbose=False):
    """(ra, dec, source) of the objects known around (ra, dec) within radius [deg]: Pan-STARRS DR2
    mean objects with >= MIN_DET detections and Gaia DR3; cache: an ECSV file reused when it
    covers the cone."""
    if cache and os.path.exists(cache):
        t = astropy.table.Table.read(cache, format='ascii.ecsv')
        m = t.meta
        if _sep(m.get('RA', np.nan), m.get('DEC', np.nan), ra, dec) / 3600 + radius <= m.get('RADIUS', -1):
            if verbose:
                print(f"deepfield: {len(t)} known objects from {cache}")
            return t
    parts, failed = [], False
    if dec > -30:
        try:
            from astroquery.mast import Catalogs
            r = Catalogs.query_region(f"{ra} {dec}", radius=radius, catalog='Panstarrs', data_release='dr2',
                                      table='mean', columns=['raMean', 'decMean', 'nDetections', 'rMeanPSFMag'],
                                      nDetections=[('gte', MIN_DET)])
            p = astropy.table.Table([np.asarray(r['raMean'], float), np.asarray(r['decMean'], float)],
                                    names=('RA', 'DEC'))
            p['MAG'] = np.where(np.asarray(r['rMeanPSFMag'], float) > 0, np.asarray(r['rMeanPSFMag'], float), np.nan)
            p['SRC'] = 'PS1'
            parts.append(p)
            if verbose:
                print(f"deepfield: Pan-STARRS DR2: {len(p)} objects (>= {MIN_DET} detections) within {radius*60:.1f}'")
        except Exception as e:
            print(f"deepfield: WARNING Pan-STARRS query failed: {e}")
            failed = True
    try:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = -1
        q = ("SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
             f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))")
        r = Gaia.launch_job_async(q, verbose=False).get_results()
        g = astropy.table.Table([np.asarray(r['ra'], float), np.asarray(r['dec'], float)], names=('RA', 'DEC'))
        g['MAG'] = np.asarray(r['phot_g_mean_mag'], float)
        g['SRC'] = 'Gaia'
        parts.append(g)
        if verbose:
            print(f"deepfield: Gaia DR3: {len(g)} objects")
    except Exception as e:
        print(f"deepfield: WARNING Gaia query failed: {e}")
        failed = True
    if not parts:
        return None
    t = astropy.table.vstack(parts)
    t.meta.update(RA=float(ra), DEC=float(dec), RADIUS=float(radius), SOURCES=sorted(set(t['SRC'])))
    if cache and not failed:      # a failed query is retried next time
        t.write(cache, format='ascii.ecsv', overwrite=True)
    return t


def frame_check(frames, ra, dec, ref_ra, ref_dec, ape_arcsec, gain=None):
    """Forced aperture photometry of (ra, dec) in every frame, the frames scaled to each other by
    the reference stars.  Returns (per-object dict of arrays, frame transparencies)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from pyrt.core.aperphot import phot
    allra = np.r_[ref_ra, ra]
    allde = np.r_[ref_dec, dec]
    nref = len(ref_ra)
    F, E = [], []
    for f in frames:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            h = fits.getheader(f)
            img = fits.getdata(f).astype(float)
            w = WCS(h)
        scale = float(np.mean(proj_plane_pixel_scales(w.celestial))) * 3600
        ape = ape_arcsec / scale
        epadu = float(gain or h.get('GAIN', 1.0) or 1.0)
        x, y = w.all_world2pix(np.c_[allra, allde], 1).T
        p = phot(img, x, y, ape, 3 * ape, 2 * ape, epadu=epadu)
        fl, sd, ar, ns = (np.asarray(p[k], float) for k in ('FLUX', 'STDEV', 'AREA', 'NSKY'))
        bad = np.asarray(p['PIER']) != 0
        with np.errstate(invalid='ignore', divide='ignore'):
            er = np.sqrt(np.clip(fl, 0, None) / epadu + ar * sd**2 + ar**2 * sd**2 / ns)
        fl[bad] = np.nan
        er[bad] = np.nan
        F.append(fl)
        E.append(er)
    F, E = np.array(F), np.array(E)
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.nanmedian(F[:, :nref] / np.nanmedian(F[:, :nref], axis=0), axis=1) if nref else np.ones(len(frames))
    rel = np.where(np.isfinite(rel) & (rel > 0), rel, np.nan)
    return F[:, nref:] / rel[:, None], E[:, nref:] / rel[:, None], rel


def frame_stats(f, e):
    """S/N of the frames together, chi2/ndf against constant, the best frame's share of the weighted
    flux, the S/N without it and the number of frames above 3 sigma."""
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    f, e = f[ok], e[ok]
    if len(f) < 2:
        return dict(SN_FRAMES=np.nan, CHI2_FRAMES=np.nan, MAX_SHARE=np.nan, SN_WO_MAX=np.nan, N_3SIG=0, N_FRAMES=len(f))
    w = 1 / e**2
    m = np.sum(w * f) / np.sum(w)
    sn = m * np.sqrt(np.sum(w))
    chi = np.sum(((f - m) / e)**2) / (len(f) - 1)
    j = int(np.argmax(f / e))
    o = np.arange(len(f)) != j
    sn2 = np.sum(w[o] * f[o]) / np.sum(w[o]) * np.sqrt(np.sum(w[o]))
    pos = np.sum(np.clip(f, 0, None) * w)
    share = f[j] * w[j] / pos if pos > 0 else np.nan
    return dict(SN_FRAMES=float(sn), CHI2_FRAMES=float(chi), MAX_SHARE=float(share), SN_WO_MAX=float(sn2),
                N_3SIG=int(np.sum(f / e > 3)), N_FRAMES=int(len(f)))


def deep_search(deep, frames=(), trigger=None, known=None, verbose=False):
    """deep: the coadd's calibrated catalogue (phcat + cat2det + dophot ECSV); frames: the FITS
    frames it was made of (their own WCS); trigger: grbinfo record or None; known: known_sources()
    output or None (then no match: everything is a candidate).  Returns an astropy Table of
    the candidates, best first."""
    ra = np.asarray(deep['ALPHA_J2000'], float)
    de = np.asarray(deep['DELTA_J2000'], float)
    mag = np.asarray(deep['MAG_CALIB'], float)
    err = np.asarray(deep['MAGERR_CALIB'], float)
    fwhm = np.asarray(deep['FWHM_IMAGE'], float) if 'FWHM_IMAGE' in deep.colnames else np.full(len(deep), np.nan)
    dkn = np.full(len(deep), np.inf)
    if known is not None and len(known):
        from scipy.spatial import cKDTree
        c0 = np.cos(np.radians(np.median(de)))
        tree = cKDTree(np.c_[np.asarray(known['RA'], float) * c0, np.asarray(known['DEC'], float)])
        dkn, _ = tree.query(np.c_[ra * c0, de])
        dkn = dkn * 3600
    isknown = dkn < MATCH
    stars = isknown & np.isfinite(fwhm) & (fwhm > 0)
    fwhm0 = float(np.median(fwhm[stars])) if np.sum(stars) >= 5 else float(deep.meta.get('FWHM', np.nan))
    extended = np.isfinite(fwhm) & (fwhm > EXT_FWHM * fwhm0)
    # phcat's forced measurement at the RTS2 target position (NUMBER 0) is not a detection
    forced = np.asarray(deep['NUMBER']) == 0 if 'NUMBER' in deep.colnames else np.zeros(len(deep), bool)
    cand = np.flatnonzero(~isknown & ~forced)
    if verbose:
        print(f"deepfield: {len(deep)} objects, {np.sum(isknown)} known (within {MATCH}\"), {len(cand)} candidates"
              f" ({np.sum(extended[cand])} extended)")
    out = astropy.table.Table()
    out['RA'], out['DEC'] = ra[cand], de[cand]
    out['MAG'], out['MAGERR'] = mag[cand], err[cand]
    out['FWHM'] = fwhm[cand]
    out['KNOWN_DIST'] = np.where(np.isfinite(dkn[cand]), dkn[cand], np.nan)
    for k in ('SN_FRAMES', 'CHI2_FRAMES', 'MAX_SHARE', 'SN_WO_MAX'):
        out[k] = np.full(len(cand), np.nan)
    out['N_3SIG'] = np.zeros(len(cand), int)
    out['N_FRAMES'] = np.zeros(len(cand), int)
    if len(frames) and len(cand):
        ref = np.flatnonzero(isknown & (mag > REF_MAG[0]) & (mag < REF_MAG[1]) & ~extended & (np.asarray(deep['FLAGS']) == 0
                             if 'FLAGS' in deep.colnames else True))
        scale = deep.meta.get('PIXEL') or deep.meta.get('PIXSCALE')
        if not scale:
            from astropy.wcs import WCS
            from astropy.wcs.utils import proj_plane_pixel_scales
            try:
                scale = float(np.mean(proj_plane_pixel_scales(WCS(deep.meta).celestial))) * 3600
            except Exception:
                scale = 1.0
        ape_arcsec = float(deep.meta.get('APERTURE', 2 * fwhm0)) * float(scale)
        F, E, rel = frame_check(frames, ra[cand], de[cand], ra[ref], de[ref], ape_arcsec)
        if verbose:
            print(f"deepfield: {len(frames)} frames, {len(ref)} reference stars, transparency "
                  + " ".join("%.2f" % r for r in rel))
        for k in range(len(cand)):
            for key, v in frame_stats(F[:, k], E[:, k]).items():
                out[key][k] = v
    cls = []
    for r, ext in zip(out, extended[cand]):
        if ext:
            cls.append('EXTENDED')
        elif not np.isfinite(r['SN_FRAMES']):
            cls.append('UNCHECKED')
        elif r['SN_FRAMES'] < SN_FRAMES:
            cls.append('FAINT')
        elif r['SN_WO_MAX'] < SN_FRAMES:
            cls.append('ONEFRAME')
        elif r['CHI2_FRAMES'] > CHI2_CONST:
            cls.append('VARIABLE')
        else:
            cls.append('NEW')
    out['CLASS'] = cls
    if trigger is not None and np.isfinite(trigger.get('ra', np.nan)):
        from pyrt.utils.grbinfo import errorbox_distance
        d = np.array([errorbox_distance(trigger, float(a), float(b)) for a, b in zip(out['RA'], out['DEC'])]).reshape(-1, 2)
        out['DIST'], out['DIST_BOX'] = d[:, 0] * 60, d[:, 1]
    else:
        out['DIST'] = out['DIST_BOX'] = np.full(len(out), np.nan)
    rank = {'NEW': 0, 'VARIABLE': 1, 'UNCHECKED': 2, 'FAINT': 3, 'ONEFRAME': 4, 'EXTENDED': 5}
    inbox = np.where(np.isfinite(out['DIST_BOX']), out['DIST_BOX'] <= 1, False)
    key = np.lexsort((-np.nan_to_num(out['SN_FRAMES'], nan=0), np.nan_to_num(out['DIST_BOX'], nan=1e9),
                      [rank[c] for c in out['CLASS']], ~inbox))
    out = out[key]
    out.meta['FWHM_STARS'] = fwhm0
    out.meta['N_KNOWN'] = int(np.sum(isknown))
    out.meta['N_OBJECTS'] = len(deep)
    return out


def report(res, deep, trigger=None, top=10):
    """The text block for the log/notifications."""
    lim = deep.meta.get('MAGZERO', np.nan) + deep.meta.get('LIMFLX3', np.nan)
    lines = ["*** DEEPFIELD START ***",
             f"deep image: {res.meta['N_OBJECTS']} objects, {res.meta['N_KNOWN']} known, limit {lim:.2f}"
             + (f", {trigger['name']} box {trigger['errorbox']*60:.2f}'" if trigger is not None and trigger.get('name') else "")]
    counts = {c: int(np.sum(res['CLASS'] == c)) for c in ('NEW', 'VARIABLE', 'ONEFRAME', 'FAINT', 'EXTENDED', 'UNCHECKED')}
    lines.append("candidates: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    for r in res[:top]:
        lines.append("%-9s %.5f %+.5f mag %5.2f±%.2f  frames S/N %4.1f chi2 %5.2f max %3.0f%% w/o %4.1f n3 %2d"
                     "  | %5.2f' = %5.2f box%s"
                     % (r['CLASS'], r['RA'], r['DEC'], r['MAG'], r['MAGERR'], r['SN_FRAMES'], r['CHI2_FRAMES'],
                        100 * r['MAX_SHARE'] if np.isfinite(r['MAX_SHARE']) else np.nan, r['SN_WO_MAX'], r['N_3SIG'],
                        r['DIST'], r['DIST_BOX'],
                        "" if not np.isfinite(r['KNOWN_DIST']) else "  (known obj. %.1f\")" % r['KNOWN_DIST']))
    lines.append("*** DEEPFIELD END ***")
    return "\n".join(lines)
