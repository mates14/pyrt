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

  - sharpness on the deep image: a pixel defect left by the dark stays on one sky position as
    long as the pointing does, and passes the frame check; it is much sharper than the PSF
  - chip or sky: stacks of the frames registered on the sky and on the chip; a coadded bad
    pixel stays on the chip as the field drifts
  - the difference against a template (Pan-STARRS, galaxy_phcat.prepare_and_subtract): an
    object present in the template is removed there

Classes, best first: NEW (in the frames throughout, consistent with constant, kept in the
difference image), VARIABLE (in the frames, but not constant), IN_TEMPLATE (removed by the
subtraction, or too faint there), FAINT (below S/N 3 in the frames together), ONEFRAME
(artefact of one frame), SHARP (pixel defect, cosmic ray), EXTENDED, MOVING (centroid shifts
between the first and second half of the frames), MINOR_PLANET (on a SkyBoT track); candidates within the
error box first, then by distance in box radii and S/N."""

import os
import warnings

import numpy as np
import astropy.table

MIN_DET = 3          # Pan-STARRS objects with fewer detections are mostly single-epoch junk
MATCH = 2.0          # [arcsec] a deep source this close to a known object is that object ...
SSO_MATCH = 10.0     # [arcsec] a candidate this close to a known minor planet's track (SkyBoT) is it
MOVE_PX = 1.0        # [px] centroid shift between the first and the second half of the frames ...
MOVE_SIG = 4.0       # ... at this significance, and not the chip's drift: a moving object (an unknown minor planet)
MOVE_FLOOR = 0.3     # [px] registration error of the half stacks
MATCH_FWHM = 0.5     # ... or within this many stars' FWHM, if more (centroids scatter with the seeing)
EXT_FWHM = 1.8       # FWHM above this times the stars' median: extended
SN_FRAMES = 3.0      # S/N of the frames together (and without the best one) to be "in the frames"
CHI2_CONST = 3.0     # chi2/ndf of the frame fluxes above this: VARIABLE
REF_MAG = (13.0, 17.5)
SHARP_RATIO = 1.5    # concentration (flux in 1 px / in the aperture) above this times the stars': SHARP
SHARP_SIG = 2.0      # ... and the core this many sigma above a star's of the same flux
DIFF_SN = 3.0        # a source in the template counts as NEW only if this significant in the difference ...
DIFF_KEEP = 0.5      # ... and keeping this fraction of its deep flux (brighter than in Pan-STARRS)
DIFF_WARN = 5.0      # below this S/N in the difference image: a caution
TEMPLATE_FRAC = 0.5  # the Pan-STARRS master holding this fraction of the deep flux: IN_TEMPLATE
MASKED = 0.1         # more of the aperture than this masked by hotpants: the difference image not used
CHIP_Z = 3.0         # sky- minus chip-registered core flux below -CHIP_Z sigma: a DEFECT (stays on the chip)
CHIP_DRIFT = 2.5     # [px] the frames' drift on the chip needed for that test
CHIP_REL = 0.6       # frames at least this transparent are used for it
NRAND = 400          # random apertures for the empirical noise of an image (correlated by resampling)


def _sep(ra1, de1, ra2, de2):
    """[arcsec], small angles"""
    return np.hypot((ra1 - ra2) * np.cos(np.radians(de1)), de1 - de2) * 3600


def minor_planets(ra, dec, radius, jd, cache=None, location='500', verbose=False):
    """The known minor planets and comets in the field at the time jd (IMCCE SkyBoT; location: the
    MPC observatory code, D50 557): table NAME, RA, DEC, RATE_RA, RATE_DEC ["/h, RA*cos(dec)], V.
    cache: an ECSV file reused for the same field and time."""
    if cache and os.path.exists(cache):
        t = astropy.table.Table.read(cache, format='ascii.ecsv')
        m = t.meta
        if (abs(m.get('JD', 0) - jd) < 1e-3 and _sep(m.get('RA', np.nan), m.get('DEC', np.nan), ra, dec) / 3600 + radius
                <= m.get('RADIUS', -1)):
            return t
    try:
        from astroquery.imcce import Skybot
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import astropy.units as u
        r = Skybot.cone_search(SkyCoord(ra, dec, unit='deg'), radius * u.deg, Time(jd, format='jd'), location=location)
        val = lambda c, unit=None: np.array([float(getattr(v, 'value', v)) for v in (r[c].to(unit) if unit else r[c])])
        t = astropy.table.Table([np.asarray(r['Name']).astype(str), val('RA'), val('DEC'),
                                 val('RA_rate', u.arcsec / u.h), val('DEC_rate', u.arcsec / u.h), val('V')],
                                names=('NAME', 'RA', 'DEC', 'RATE_RA', 'RATE_DEC', 'V'))
    except Exception as e:
        if 'No solar system object' in str(e) or 'empty' in str(e).lower():
            t = astropy.table.Table(names=('NAME', 'RA', 'DEC', 'RATE_RA', 'RATE_DEC', 'V'), dtype=(str, float, float, float, float, float))
        else:
            print(f"deepfield: WARNING SkyBoT query failed: {e}")
            return None
    t.meta.update(RA=float(ra), DEC=float(dec), RADIUS=float(radius), JD=float(jd))
    if verbose:
        print(f"deepfield: SkyBoT: {len(t)} known minor planets/comets in the field" +
              (", " + ", ".join(f"{n} V={v:.1f}" for n, v in zip(t['NAME'], t['V']) if v < 21) if len(t) else ""))
    if cache:
        t.write(cache, format='ascii.ecsv', overwrite=True)
    return t


def _sso_match(sso, ra, dec, hours):
    """Name of the known minor planet whose track over the stack (hours long, centred at the
    query time) passes within SSO_MATCH of each (ra, dec), or ''."""
    out = np.full(len(ra), '', dtype=object)
    if sso is None or not len(sso):
        return out
    c = np.cos(np.radians(np.asarray(dec, float)))
    for s in sso:
        tt = np.linspace(-hours / 2, hours / 2, 21)
        tr = s['RA'] + tt * s['RATE_RA'] / 3600 / np.cos(np.radians(s['DEC']))
        td = s['DEC'] + tt * s['RATE_DEC'] / 3600
        d = np.min(np.hypot((np.asarray(ra, float)[:, None] - tr[None]) * c[:, None],
                            np.asarray(dec, float)[:, None] - td[None]), axis=1) * 3600
        out[(d < SSO_MATCH) & (out == '')] = s['NAME']
    return out


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
    if failed:
        print(f"deepfield: the known objects are from {', '.join(t.meta['SOURCES'])} only")
    if cache:                     # what was had; delete the file to query again
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


def _aperture_noise(img, ape, seed=1):
    """Robust sigma of the flux in NRAND random apertures: the noise of an aperture with the
    pixel correlation of a resampled (coadded, subtracted) image, which the annulus stdev misses."""
    from pyrt.core.aperphot import phot
    rng = np.random.default_rng(seed)
    ny, nx = img.shape
    m = int(4 * ape) + 2
    x, y = rng.uniform(m, nx - m, NRAND), rng.uniform(m, ny - m, NRAND)
    f = np.asarray(phot(img, x, y, ape, 3 * ape, 2 * ape, epadu=1e9)['FLUX'], float)
    f = f[np.isfinite(f)]
    return 1.4826 * float(np.median(np.abs(f - np.median(f)))) if len(f) > 20 else np.nan


def chip_or_sky(frames, ra, dec, rel, h=7):
    """Does a source follow the sky or stay on the chip?  The frames (transparency >= CHIP_REL)
    stacked twice around each position: shifted so that the sky position is at the centre, and
    cut at one fixed chip pixel.  A source on the sky is sharper in the first, a pixel defect
    (left by the dark, coadded where the pointing stood) in the second; z = (core flux sky-
    registered - chip-registered) / its noise.  Returns z per position and the drift [px] of the
    field over the frames used (GRB 261113.604 at D50, r: 38 frames, 6.5 x 4.7 px drift, the
    afterglow +6.5, warm pixels -5.5 to -13.7)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import shift as nshift
    use = [f for f, r in zip(frames, rel) if np.isfinite(r) and r >= CHIP_REL]
    z = np.full(len(ra), np.nan)
    mv, mvs = np.full(len(ra), np.nan), np.full(len(ra), np.nan)
    if len(use) < 4 or not len(ra):
        return z, 0.0, mv, mvs
    P, imgs = [], []
    for f in use:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            P.append(WCS(fits.getheader(f)).all_world2pix(np.c_[ra, dec], 0))
            imgs.append(fits.getdata(f).astype(float))
    P = np.array(P)
    drift = float(np.median(np.hypot(np.ptp(P[:, :, 0], axis=0), np.ptp(P[:, :, 1], axis=0))))
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    rr = np.hypot(xx, yy)
    core, ring = rr <= 1.5, (rr > 4) & (rr <= h)
    for k in range(len(ra)):
        q = np.round(np.median(P[:, k], axis=0)).astype(int)
        S, C, used = [], [], []
        for i, img in enumerate(imgs):
            x, y = P[i, k]
            ix, iy = int(np.floor(x)), int(np.floor(y))
            b = img[iy - h - 1:iy + h + 2, ix - h - 1:ix + h + 2]
            cb = img[q[1] - h:q[1] + h + 1, q[0] - h:q[0] + h + 1]
            if b.shape != (2 * h + 3, 2 * h + 3) or cb.shape != (2 * h + 1, 2 * h + 1) or min(iy - h - 1, ix - h - 1, q[0] - h, q[1] - h) < 0:
                continue
            sb = nshift(b, (-(y - iy), -(x - ix)), order=1)[1:-1, 1:-1]
            used.append(i)
            S.append(sb - np.median(sb[ring]))
            C.append(cb - np.median(cb[ring]))
        if len(S) < 4:
            continue
        v, e = _halves_shift(S, core | (rr <= 3), ring)
        Pk = P[used, k]
        n2 = len(Pk) // 2
        chip = -(Pk[n2:].mean(axis=0) - Pk[:n2].mean(axis=0))   # a pixel fixed on the chip shifts so
        mv[k] = float(np.hypot(*v))
        # significant, and not the chip drift (a coadded bad pixel moves like that in sky stacks)
        mvs[k] = min(mv[k], float(np.hypot(*(v - chip)))) / e if e > 0 else np.nan
        S, C = np.mean(S, axis=0), np.mean(C, axis=0)
        bg = np.r_[S[ring], C[ring]]
        e = np.sqrt(core.sum()) * 1.4826 * np.median(np.abs(bg - np.median(bg)))
        z[k] = (S[core].sum() - C[core].sum()) / e if e > 0 else np.nan
    return z, drift, mv, mvs


def _halves_shift(S, cen, ring):
    """Centroid shift (dx, dy) [px] between the sky-registered stacks of the first and the second
    half of the frames (in time order) and its error (with MOVE_FLOOR for the registration): a
    minor planet moves, a star does not."""
    n = len(S) // 2
    yy, xx = np.mgrid[:S[0].shape[0], :S[0].shape[1]]
    c, sig = [], []
    for part in (S[:n], S[n:]):
        a = np.mean(part, axis=0)
        bg = a[ring]
        noise = 1.4826 * np.median(np.abs(bg - np.median(bg)))
        w = np.clip(a, 0, None) * cen
        f = w.sum()
        if f <= 0 or noise <= 0:
            return np.array([np.nan, np.nan]), np.nan
        x, y = (w * xx).sum() / f, (w * yy).sum() / f
        m2 = np.sqrt(max((w * ((xx - x) ** 2 + (yy - y) ** 2)).sum() / f / 2, 0.25))
        sig.append(m2 * noise * np.sqrt(cen.sum()) / f)      # centroid error ~ size / (S/N)
        c.append((x, y))
    return np.array([c[1][0] - c[0][0], c[1][1] - c[0][1]]), float(np.hypot(np.hypot(*sig), MOVE_FLOOR))


def template_frac(template, ra, dec, mag, ape_arcsec):
    """The Pan-STARRS master (ps1mosaic: counts/s, AB zero point PS1ZP) measured at each position in
    the deep image's own aperture: the fraction of the deep flux that is there (from MAG, in the
    catalogue's PS1-like system) and its S/N against random apertures (the stack's noise is
    correlated).  The master is much deeper than a coadd, but a faint galaxy is spread: in a 2"
    aperture it was 'nothing', in the 4.5" one it had the deep magnitude (GRB 261113.604 in g,
    three such; the afterglow had nothing at any aperture)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from pyrt.core.aperphot import phot
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = fits.getheader(template)
        w = WCS(h)
        x, y = w.all_world2pix(np.c_[ra, dec], 1).T
        T = fits.getdata(template).astype(float)
    ape = ape_arcsec / (float(np.mean(proj_plane_pixel_scales(w.celestial))) * 3600)
    m = 4 * ape
    inside = (x > m) & (y > m) & (x < T.shape[1] - m) & (y < T.shape[0] - m)
    frac, sn = np.full(len(ra), np.nan), np.full(len(ra), np.nan)
    if not np.any(inside):
        return frac, sn
    f = np.asarray(phot(T, x[inside], y[inside], ape, 3 * ape, 2 * ape, epadu=1e9)['FLUX'], float)
    rng = np.random.default_rng(1)
    lo_x, hi_x = max(m, np.min(x[inside]) - 200), min(T.shape[1] - m, np.max(x[inside]) + 200)
    lo_y, hi_y = max(m, np.min(y[inside]) - 200), min(T.shape[0] - m, np.max(y[inside]) + 200)
    q = np.asarray(phot(T, rng.uniform(lo_x, hi_x, NRAND), rng.uniform(lo_y, hi_y, NRAND), ape, 3 * ape, 2 * ape,
                        epadu=1e9)['FLUX'], float)
    q = q[np.isfinite(q)]
    noise = 1.4826 * np.median(np.abs(q - np.median(q))) if len(q) > 20 else np.nan
    expect = 10**(-0.4 * (np.asarray(mag, float)[inside] - h.get('PS1ZP', 25.0)))
    frac[inside], sn[inside] = f / expect, f / noise
    return frac, sn


def image_checks(image, ra, dec, ref_ra, ref_dec, ape, diff=None):
    """On the deep image: the concentration C = flux(r = 1 px) / flux(aperture) of each position
    and of the reference stars (a hot pixel left by the dark or a cosmic ray is much sharper than
    the PSF: GRB 261113.604 at D50, the pointing still for 35 of 37 frames, seven warm pixels
    stacked into steady "sources" at C 0.36-0.58, stars 0.21-0.235, the afterglow 0.29), the core's
    excess over a star's [sigma]; on the difference image (if given): the flux kept and its S/N."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from pyrt.core.aperphot import phot
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        D = fits.getdata(image).astype(float)
        w = WCS(fits.getheader(image))
    allra, allde = np.r_[ref_ra, ra], np.r_[ref_dec, dec]
    x, y = w.all_world2pix(np.c_[allra, allde], 1).T
    a = phot(D, x, y, 1.0, 3 * ape, 2 * ape, epadu=1e9)
    b = phot(D, x, y, ape, 3 * ape, 2 * ape, epadu=1e9)
    fa, fb = np.asarray(a['FLUX'], float), np.asarray(b['FLUX'], float)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = fa / fb
    n = len(ref_ra)
    c0 = float(np.nanmedian(C[:n])) if n >= 5 else np.nan
    ecore = _aperture_noise(D, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        core_sig = (fa[n:] - c0 * fb[n:]) / ecore
    out = dict(CONC=C[n:], CORE_SIG=core_sig, DEEP_FLUX=fb[n:])
    if diff is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            S = fits.getdata(diff).astype(float)
        fs = np.asarray(phot(S, x[n:], y[n:], ape, 3 * ape, 2 * ape, epadu=1e9)['FLUX'], float)
        # hotpants fills what it masks with ~1e-30: the fraction of each aperture in the mask
        mk = (np.abs(S) < 1e-20).astype(float)
        fm = np.asarray(phot(mk, x[n:], y[n:], ape, 3 * ape, 2 * ape, epadu=1e9)['SUM'], float) / (np.pi * ape**2)
        es = _aperture_noise(S, ape)
        with np.errstate(invalid='ignore', divide='ignore'):
            out['DIFF_SN'] = np.where(fm > MASKED, np.nan, fs / es)
            out['DIFF_KEEP'] = np.where(fm > MASKED, np.nan, fs / fb[n:])
    return out, c0


def _pixscale(deep):
    """The deep image's pixel scale [arcsec]: its PIXEL/PIXSCALE keyword, or from its WCS"""
    sc = deep.meta.get('PIXEL') or deep.meta.get('PIXSCALE')
    if sc:
        return float(sc)
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    try:
        return float(np.mean(proj_plane_pixel_scales(WCS(deep.meta).celestial))) * 3600
    except Exception:
        return 1.0


def deep_search(deep, frames=(), trigger=None, known=None, image=None, diff=None, template=None, sso=None,
                verbose=False):
    """deep: the coadd's calibrated catalogue (phcat + cat2det + dophot ECSV); frames: the FITS
    frames it was made of (their own WCS); trigger: grbinfo record or None; known: known_sources()
    output or None (then no match: everything is a candidate); image: the deep FITS (the
    sharpness test); diff: its difference against a template (galaxy_phcat.prepare_and_subtract).
    sso: minor_planets() output or None.  Returns an astropy Table of the candidates, best first."""
    frames = sorted(frames, key=os.path.basename)    # time order (the names start with it)
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
    sc = _pixscale(deep)
    stars = (dkn < MATCH) & np.isfinite(fwhm) & (fwhm > 0)
    fwhm0 = float(np.median(fwhm[stars])) if np.sum(stars) >= 5 else float(deep.meta.get('FWHM', np.nan))
    match = max(MATCH, MATCH_FWHM * fwhm0 * sc) if np.isfinite(fwhm0) else MATCH
    isknown = dkn < match
    extended = np.isfinite(fwhm) & (fwhm > EXT_FWHM * fwhm0)
    # phcat's forced measurement at the RTS2 target position (NUMBER 0) is not a detection
    forced = np.asarray(deep['NUMBER']) == 0 if 'NUMBER' in deep.colnames else np.zeros(len(deep), bool)
    cand = np.flatnonzero(~isknown & ~forced)
    if verbose:
        print(f"deepfield: {len(deep)} objects, {np.sum(isknown)} known (within {match:.1f}\"), {len(cand)} candidates"
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
    for k in ('CONC', 'CORE_SIG', 'DIFF_SN', 'DIFF_KEEP', 'CHIP_Z', 'MOVE', 'MOVE_SIG', 'TEMPLATE_FRAC', 'TEMPLATE_SN'):
        out[k] = np.full(len(cand), np.nan)
    hours = (float(deep.meta.get('EXPEND', 0)) - float(deep.meta.get('EXPSTART', 0))) * 24
    out['SSO'] = _sso_match(sso, ra[cand], de[cand], hours if 0 < hours < 48 else 0.0).astype(str)
    if template is not None and len(cand):
        tf, ts = template_frac(template, ra[cand], de[cand], mag[cand], float(deep.meta.get('APERTURE', 2 * fwhm0)) * sc)
        out['TEMPLATE_FRAC'], out['TEMPLATE_SN'] = tf, ts
    chip_drift = 0.0
    ref = np.flatnonzero(isknown & (mag > REF_MAG[0]) & (mag < REF_MAG[1]) & ~extended & (np.asarray(deep['FLAGS']) == 0
                         if 'FLAGS' in deep.colnames else True))
    conc0 = np.nan
    if image is not None and len(cand):
        ic, conc0 = image_checks(image, ra[cand], de[cand], ra[ref], de[ref], float(deep.meta.get('APERTURE', 2 * fwhm0)), diff)
        for k, v in ic.items():
            if k in out.colnames:
                out[k] = v
        if verbose:
            print(f"deepfield: stars' concentration {conc0:.3f}" + (f", difference image {diff}" if diff else ""))
    if len(frames) and len(cand):
        ape_arcsec = float(deep.meta.get('APERTURE', 2 * fwhm0)) * sc
        F, E, rel = frame_check(frames, ra[cand], de[cand], ra[ref], de[ref], ape_arcsec)
        if verbose:
            print(f"deepfield: {len(frames)} frames, {len(ref)} reference stars, transparency "
                  + " ".join("%.2f" % r for r in rel))
        for k in range(len(cand)):
            for key, v in frame_stats(F[:, k], E[:, k]).items():
                out[key][k] = v
        notext = ~extended[cand]
        z, drift, mv, mvs = chip_or_sky(frames, ra[cand], de[cand], rel)
        out['CHIP_Z'][notext] = z[notext]          # a galaxy is no coadded bad pixel
        out['MOVE'], out['MOVE_SIG'] = mv, mvs
        chip_drift = drift
        if verbose:
            print(f"deepfield: chip-or-sky test: the field drifted {drift:.1f} px on the chip over the clear frames"
                  + ("" if drift >= CHIP_DRIFT else f" (< {CHIP_DRIFT} px: not usable)"))
    cls = []
    for r, ext in zip(out, extended[cand]):
        if r['SSO']:
            cls.append('MINOR_PLANET')
        elif ext:
            cls.append('EXTENDED')
        elif chip_drift >= CHIP_DRIFT and r['CHIP_Z'] < -CHIP_Z:
            cls.append('DEFECT')
        elif np.isfinite(conc0) and r['CONC'] > SHARP_RATIO * conc0 and r['CORE_SIG'] > SHARP_SIG:
            cls.append('SHARP')
        elif not np.isfinite(r['SN_FRAMES']):
            cls.append('UNCHECKED')
        elif r['SN_FRAMES'] < SN_FRAMES:
            cls.append('FAINT')
        elif r['SN_WO_MAX'] < SN_FRAMES:
            cls.append('ONEFRAME')
        elif _in_template(r):
            cls.append('IN_TEMPLATE')
        elif r['MOVE'] > MOVE_PX and r['MOVE_SIG'] > MOVE_SIG:
            # last: galaxies and the coadded bad pixels shift between the half stacks too
            cls.append('MOVING')
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
    rank = {'NEW': 0, 'VARIABLE': 1, 'UNCHECKED': 2, 'IN_TEMPLATE': 3, 'FAINT': 4, 'ONEFRAME': 5, 'SHARP': 6, 'DEFECT': 7, 'EXTENDED': 8,
            'MOVING': 9, 'MINOR_PLANET': 10}
    inbox = np.where(np.isfinite(out['DIST_BOX']), out['DIST_BOX'] <= 1, False)
    key = np.lexsort((-np.nan_to_num(out['SN_FRAMES'], nan=0), np.nan_to_num(out['DIST_BOX'], nan=1e9),
                      [rank[c] for c in out['CLASS']], ~inbox))
    out = out[key]
    out.meta['FWHM_STARS'] = fwhm0
    out.meta['CONC_STARS'] = conc0
    out.meta['CHIP_DRIFT'] = chip_drift
    out.meta['N_KNOWN'] = int(np.sum(isknown))
    out.meta['N_OBJECTS'] = len(deep)
    return out


NEAR_BOX = 2.0       # [box radii] a NEW this close to the trigger position is an afterglow candidate

REJECT = {'IN_TEMPLATE': 'in the Pan-STARRS image', 'FAINT': 'too faint in the frames',
          'ONEFRAME': 'from one frame only (satellite, aircraft, cosmic ray)',
          'SHARP': 'sharper than a star (pixel defect, cosmic ray)',
          'DEFECT': 'staying on the chip, not on the sky (coadded bad pixels)', 'EXTENDED': 'extended (galaxies)',
          'VARIABLE': 'variable in the frames', 'UNCHECKED': 'not checked in the frames',
          'MOVING': 'moving between the first and second half of the frames (minor planets)',
          'MINOR_PLANET': 'known minor planets (SkyBoT)'}


def _in_template(r):
    """In the template and not (significantly) brighter now.  With the master measured, that
    decides; otherwise the difference image alone: a source removed (keeps < DIFF_KEEP) while
    measurably so."""
    brighter = np.isfinite(r['DIFF_SN']) and r['DIFF_SN'] >= DIFF_SN and r['DIFF_KEEP'] >= DIFF_KEEP
    if np.isfinite(r['TEMPLATE_FRAC']):
        return r['TEMPLATE_FRAC'] >= TEMPLATE_FRAC and not brighter
    if np.isfinite(r['DIFF_SN']) and np.isfinite(r['DIFF_KEEP']) and r['DIFF_KEEP'] < DIFF_KEEP:
        err = r['DIFF_KEEP'] / r['DIFF_SN'] if r['DIFF_SN'] != 0 else np.inf
        return r['DIFF_KEEP'] + 2 * abs(err) < 1
    return False


def _why(r, conc0, drift=0.0):
    """Plain words for a NEW, with the tests it passed only narrowly."""
    words = []
    if drift >= CHIP_DRIFT and np.isfinite(r['CHIP_Z']):
        words.append(f"follows the sky, not the chip ({r['CHIP_Z']:+.1f} sigma over a {drift:.1f} px drift)")
    if np.isfinite(r['SN_FRAMES']):
        words.append(f"in the frames throughout (S/N {r['SN_FRAMES']:.1f}, at most {100*r['MAX_SHARE']:.0f}% from one)")
    if np.isfinite(r['TEMPLATE_FRAC']):
        words.append(f"Pan-STARRS image has {100*max(r['TEMPLATE_FRAC'], 0):.0f}% of its flux (S/N {r['TEMPLATE_SN']:.1f})"
                     if r['TEMPLATE_FRAC'] < TEMPLATE_FRAC else
                     f"brighter than in Pan-STARRS (which has {100*r['TEMPLATE_FRAC']:.0f}% of it)")
    if np.isfinite(r['DIFF_KEEP']):
        words.append(f"kept {100*r['DIFF_KEEP']:.0f}% after subtracting it (S/N {r['DIFF_SN']:.1f})")
    elif np.isfinite(r['KNOWN_DIST']):
        words.append(f"no Pan-STARRS/Gaia object within {r['KNOWN_DIST']:.0f}\"")
    warn = []
    if drift < CHIP_DRIFT:
        warn.append(f"the field drifted only {drift:.1f} px on the chip: a coadded bad pixel cannot be excluded")
    elif np.isfinite(r['CHIP_Z']) and r['CHIP_Z'] < 2:
        warn.append(f"undecided whether on the sky or on the chip ({r['CHIP_Z']:+.1f} sigma)")
    if np.isfinite(conc0) and np.isfinite(r['CONC']) and r['CONC'] > 0.8 * SHARP_RATIO * conc0:
        warn.append(f"nearly as sharp as a pixel defect (C {r['CONC']:.2f}, stars {conc0:.2f}, cut {SHARP_RATIO*conc0:.2f})")
    if np.isfinite(r['DIFF_SN']) and r['DIFF_SN'] < DIFF_WARN:
        warn.append(f"only S/N {r['DIFF_SN']:.1f} in the difference image")
    if np.isfinite(r['SN_FRAMES']) and r['SN_FRAMES'] < SN_FRAMES + 2:
        warn.append(f"faint in the frames (S/N {r['SN_FRAMES']:.1f})")
    return "; ".join(words), warn


def report(res, deep, trigger=None, top=10, verbose=False):
    """The text block for the log/notifications: the verdict, the NEW objects in words, what
    was rejected and why; verbose: the whole table."""
    lim = deep.meta.get('MAGZERO', np.nan) + deep.meta.get('LIMFLX3', np.nan)
    band = str(deep.meta.get('FILTER', '')).replace('Sloan_', '').replace('Sloan-', '') or 'mag'
    nfr = int(np.max(res['N_FRAMES'])) if len(res) and np.max(res['N_FRAMES']) > 0 else None
    conc0 = res.meta.get('CONC_STARS', np.nan)
    lines = ["*** DEEPFIELD START ***"]
    head = []
    if trigger is not None and trigger.get('name'):
        head.append(f"{trigger['name']}, error box {trigger['errorbox']*60:.2f}'")
    head.append(f"deep {band} image" + (f" of {nfr} frames" if nfr else "") + f", limit {band} = {lim:.2f}")
    lines.append("; ".join(head))
    new = res[res['CLASS'] == 'NEW']
    hasbox = len(res) and np.any(np.isfinite(res['DIST_BOX']))
    near = new[np.nan_to_num(new['DIST_BOX'], nan=1e9) <= NEAR_BOX] if hasbox else new[:0]
    far = new[~(np.nan_to_num(new['DIST_BOX'], nan=1e9) <= NEAR_BOX)] if hasbox else new
    if not hasbox:
        lines.append(f"VERDICT: {len(new)} new object(s), no trigger position to compare with")
    elif len(near):
        lines.append(f"VERDICT: {len(near)} AFTERGLOW CANDIDATE(S) within {NEAR_BOX:.0f} error-box radii")
    else:
        lines.append(f"VERDICT: nothing new within {NEAR_BOX:.0f} error-box radii down to {band} = {lim:.2f}")
    for r in near:
        why, warn = _why(r, conc0, res.meta.get('CHIP_DRIFT', 0.0))
        lines.append(f"  CANDIDATE {r['RA']:.5f} {r['DEC']:+.5f}  {band} = {r['MAG']:.2f} ± {r['MAGERR']:.2f}"
                     f"  {r['DIST']:.2f}' from the trigger position ({r['DIST_BOX']:.1f} box radii)")
        lines.append(f"            {why}")
        for w in warn:
            lines.append(f"            caution: {w}")
    if len(far):
        lines.append("other new objects (further out, unlikely the afterglow):" if hasbox else "new objects:")
        for r in far[:top]:
            _, warn = _why(r, conc0, res.meta.get('CHIP_DRIFT', 0.0))
            lines.append(f"  {r['RA']:.5f} {r['DEC']:+.5f}  {band} = {r['MAG']:.2f}"
                         + (f"  {r['DIST']:.1f}' ({r['DIST_BOX']:.1f} box radii)" if hasbox else "")
                         + ("  [" + "; ".join(warn) + "]" if warn else ""))
    rej = [f"{res.meta['N_KNOWN']} known (Pan-STARRS/Gaia)"]
    rej += [f"{int(np.sum(res['CLASS'] == c))} {w}" for c, w in REJECT.items() if np.sum(res['CLASS'] == c)]
    lines.append("rejected: " + ", ".join(rej))
    if verbose:
        lines.append("all candidates (class, position, mag, frames S/N chi2 max-share S/N-without-max, concentration, "
                     "difference S/N and flux kept, half-to-half shift, distance):")
        for r in res:
            lines.append("  %-12s %.5f %+.5f %5.2f±%.2f  fr %4.1f %5.2f %3.0f%% %4.1f  C %.2f  diff %4.1f %4.2f  mv %4.1fpx %4.1fσ  %5.2f' %5.2f box%s"
                         % (r['CLASS'], r['RA'], r['DEC'], r['MAG'], r['MAGERR'], r['SN_FRAMES'], r['CHI2_FRAMES'],
                            100 * r['MAX_SHARE'] if np.isfinite(r['MAX_SHARE']) else np.nan, r['SN_WO_MAX'],
                            r['CONC'], r['DIFF_SN'], r['DIFF_KEEP'], r['MOVE'], r['MOVE_SIG'], r['DIST'], r['DIST_BOX'],
                            f"  {r['SSO']}" if r['SSO'] else ""))
    lines.append("*** DEEPFIELD END ***")
    return "\n".join(lines)
