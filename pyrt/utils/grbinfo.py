"""
grbinfo - trigger (GRB/EP/...) information for an RTS2 target: T0, position, error box.

The target id is only unique within one telescope's RTS2 database, so a lookup is
(telescope, target).  Order:

  1. the cache file (hand-editable; ~/.config/pyrt/triggers.dat or $PYRT_TRIGGERS),
  2. the telescope's database over ssh (read-only psql select), the result is
     appended to the cache, so it works offline next time.

Cache format, one trigger per line, lines starting with '#' are comments, columns separated by blanks:

    # telescope target  t0 (UTC ISO)              ra[deg]   dec[deg]  errorbox[deg]  type  name...
    D50         53486   2026-11-06T21:28:37.699   46.972    24.975    0.05156901     210   EP 261106.894

  - telescope '*' matches any telescope (handy for a hand-added trigger),
  - '-' in t0 marks a target known to have no trigger (no more ssh for it),
  - errorbox is a radius in degrees as stored in the RTS2 grb table,
  - the last matching line wins, so a hand-written correction can simply be appended.

Telescopes ("sites"): a short key used in the cache, the ssh host with its RTS2 database,
and how to recognise the frames: the TELESCOP keyword or, when that is empty (FRAM ORM),
the site LATITUDE/LONGITUD (within 0.1 deg; D50 and SBT share Ondrejov, so they rely on
TELESCOP).  Built in:

    D50         d    TELESCOP 'D50'
    SBT         l    TELESCOP 'SBT'
    FRAM-Auger  ll   TELESCOP 'FRAM at Pierre Auger Observatory'  (-35.496, -69.450)
    FRAM-ORM    orm  TELESCOP ''                                   (+28.771, -17.894)

~/.config/pyrt/trigger-hosts (or $PYRT_TRIGGER_HOSTS) changes or adds sites, one per line:

    KEY host [db=stars] [telescop="..."] [site=lat,lon]

e.g.  'BOOTES-1 none telescop="BOOTES-1"' (host none: cache only, add triggers by hand),
      'D50 local' on the D50 machine itself (psql without ssh).
"""

import os
import shlex
import warnings
import subprocess
import sys

import numpy as np
from astropy.time import Time

DEFAULT_SITES = {
    'D50':        {'host': 'd',   'db': 'stars', 'telescop': 'D50', 'site': None},
    'SBT':        {'host': 'l',   'db': 'stars', 'telescop': 'SBT', 'site': None},
    'FRAM-Auger': {'host': 'll',  'db': 'stars', 'telescop': 'FRAM at Pierre Auger Observatory', 'site': (-35.4958, -69.4497)},
    'FRAM-ORM':   {'host': 'orm', 'db': 'stars', 'telescop': None, 'site': (28.7706, -17.8944)},
}
SITE_TOLERANCE = 0.1  # [deg]
SSH_TIMEOUT = 8  # [s] the network is often slow or absent at the telescopes

_memo = {}
_warned = set()


def _jd(iso):
    with warnings.catch_warnings():  # test triggers in the far future ("dubious year")
        warnings.simplefilter('ignore')
        return Time(iso, format='isot', scale='utc').jd


def cache_path():
    return os.path.expanduser(os.environ.get('PYRT_TRIGGERS', '~/.config/pyrt/triggers.dat'))


def hosts_path():
    return os.path.expanduser(os.environ.get('PYRT_TRIGGER_HOSTS', '~/.config/pyrt/trigger-hosts'))


def load_sites():
    sites = {k: dict(v) for k, v in DEFAULT_SITES.items()}
    try:
        with open(hosts_path()) as f:
            for line in f:
                if line.lstrip().startswith('#'):
                    continue
                w = shlex.split(line, comments=True)
                if len(w) < 2:
                    continue
                site = sites.setdefault(w[0], {'host': None, 'db': 'stars', 'telescop': None, 'site': None})
                site['host'] = None if w[1] in ('none', '-') else w[1]
                for kv in w[2:]:
                    k, _, v = kv.partition('=')
                    if k == 'db':
                        site['db'] = v
                    elif k == 'telescop':
                        site['telescop'] = v
                    elif k == 'site':
                        site['site'] = tuple(float(x) for x in v.split(','))
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        print("grbinfo: problem reading %s: %s" % (hosts_path(), e), file=sys.stderr)
    return sites


def site_key(meta):
    """Site key (D50, SBT, FRAM-Auger, FRAM-ORM, ...) of a frame from its header/ECSV meta, or None."""
    sites = load_sites()
    tel = str(meta.get('TELESCOP') or '').strip()
    if tel:
        for key, s in sites.items():
            if s['telescop'] and s['telescop'] == tel:
                return key
        if tel in sites:
            return tel
    try:
        lat, lon = float(meta.get('LATITUDE')), float(meta.get('LONGITUD'))
    except (TypeError, ValueError):
        return tel or None
    for key, s in sites.items():
        if s['site'] and abs(s['site'][0] - lat) < SITE_TOLERANCE and \
                abs((s['site'][1] - lon + 180) % 360 - 180) < SITE_TOLERANCE and not (tel and s['telescop']):
            return key
    return tel or None


def _parse_line(line):
    if line.lstrip().startswith('#'):  # (names contain '#', e.g. "trigger #01709322274")
        return None
    w = line.split()
    if len(w) < 2:
        return None
    rec = {'telescope': w[0], 'target': w[1], 't0': None, 'jd0': np.nan, 'ra': np.nan,
           'dec': np.nan, 'errorbox': np.nan, 'type': None, 'name': ''}
    if len(w) < 3 or w[2] == '-':
        return rec  # known not to be a trigger
    try:
        rec['t0'] = w[2]
        rec['jd0'] = _jd(w[2])
        rec['ra'], rec['dec'], rec['errorbox'] = [float(x) if x != '-' else np.nan for x in w[3:6]]
        rec['type'] = w[6] if len(w) > 6 else None
        rec['name'] = ' '.join(w[7:])
    except (ValueError, IndexError):
        print("grbinfo: cannot parse line in %s: %s" % (cache_path(), line.strip()), file=sys.stderr)
        return None
    return rec


def _from_cache(telescope, target):
    found = None
    try:
        with open(cache_path()) as f:
            for line in f:
                rec = _parse_line(line)
                if rec and rec['target'] == str(target) and rec['telescope'] in (telescope, '*'):
                    found = rec
    except OSError:
        pass
    return found


def _to_cache(rec):
    path = cache_path()
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        if new:
            f.write("# pyrt trigger cache (see pyrt/utils/grbinfo.py), edit freely; last matching line wins\n"
                    "# telescope target t0(UTC ISO) ra[deg] dec[deg] errorbox[deg] type name\n")
        if rec['t0'] is None:
            f.write("%s %s -\n" % (rec['telescope'], rec['target']))
        else:
            f.write("%s %s %s %.6f %.6f %.6g %s %s\n" % (rec['telescope'], rec['target'], rec['t0'],
                    rec['ra'], rec['dec'], rec['errorbox'], rec['type'], rec['name']))


def _from_database(telescope, target):
    site = load_sites().get(telescope)
    if site is None or not site['host']:
        return None
    host, db = site['host'], site['db']
    # read-only: a plain select (rts2-targetinfo -g may update the target)
    sql = ("select to_char(g.grb_date,'YYYY-MM-DD\"T\"HH24:MI:SS.MS'), g.grb_ra, g.grb_dec, g.grb_errorbox, "
           "g.grb_type, t.tar_name from grb g join targets t on t.tar_id=g.tar_id "
           "where g.tar_id=%d order by g.grb_last_update desc limit 1" % int(target))
    if host == 'local':  # on the telescope machine itself
        cmd = ['psql', '-X', '-At', '-F', '|', db, '-c', sql]
    else:
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=%d' % SSH_TIMEOUT, host,
               'psql -X -At -F "|" %s -c "%s"' % (db, sql.replace('"', '\\"'))]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3*SSH_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError) as e:
        out = None
        err = str(e)
    if out is None or out.returncode != 0:
        if (telescope, host) not in _warned:
            print("grbinfo: %s database at %s not reachable (%s), add the trigger to %s by hand"
                  % (telescope, host, err if out is None else out.stderr.strip()[:200], cache_path()), file=sys.stderr)
            _warned.add((telescope, host))
        return None
    rec = {'telescope': telescope, 'target': str(target), 't0': None, 'jd0': np.nan, 'ra': np.nan,
           'dec': np.nan, 'errorbox': np.nan, 'type': None, 'name': ''}
    line = out.stdout.strip().splitlines()
    if line:
        w = line[0].split('|')
        rec['t0'] = w[0]
        rec['jd0'] = _jd(w[0])
        rec['ra'], rec['dec'], rec['errorbox'] = [float(x) if x else np.nan for x in w[1:4]]
        rec['type'] = w[4]
        rec['name'] = ' '.join(w[5].split()) if len(w) > 5 else ''
    _to_cache(rec)  # also "no trigger", so that we do not ask again
    return rec


def get_trigger(target, telescope=None, meta=None):
    """Trigger record (dict: t0, jd0, ra, dec, errorbox, type, name) or None.
    A target known to be no trigger gives a record with t0=None.
    telescope: site key or TELESCOP value; meta: frame header/ECSV meta to recognise the site."""
    if meta is not None:
        target = meta.get('TARGET') if target is None else target
        telescope = site_key(meta)
    elif telescope is not None:
        telescope = site_key({'TELESCOP': telescope})
    if target is None:
        return None
    try:
        target = int(target)
    except (TypeError, ValueError):
        return None
    telescope = str(telescope or '*').strip()
    key = (telescope, target)
    if key not in _memo:
        rec = _from_cache(telescope, target)
        if rec is None:
            rec = _from_database(telescope, target)
        _memo[key] = rec
    return _memo[key]


def trigger_t0(target, telescope=None, meta=None):
    """JD of the trigger, 0 if unknown (as the old grbt0-based helper)."""
    rec = get_trigger(target, telescope, meta)
    return rec['jd0'] if rec and np.isfinite(rec['jd0']) else 0


def trigger_name(target, telescope=None, meta=None):
    rec = get_trigger(target, telescope, meta)
    return rec['name'] if rec and rec['name'] else '-'


def errorbox_distance(rec, ra, dec):
    """Distance [deg] of (ra, dec) from the trigger position and the same in error-box radii."""
    if rec is None or not np.isfinite(rec['ra']):
        return np.nan, np.nan
    r1, d1, r2, d2 = np.radians([rec['ra'], rec['dec'], ra, dec])
    dist = np.degrees(np.arccos(np.clip(np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2), -1, 1)))
    return dist, dist/rec['errorbox'] if rec['errorbox'] > 0 else np.nan


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Trigger info for an RTS2 target (cached).")
    ap.add_argument('target', help="target id, or a FITS/ECSV frame (target and site from its header)")
    ap.add_argument('-t', '--telescope', default='D50', help="site key or TELESCOP (default D50): " + ", ".join(load_sites()))
    a = ap.parse_args()
    if os.path.isfile(a.target):
        if a.target.endswith('.ecsv'):
            from astropy.table import Table
            meta = Table.read(a.target).meta
        else:
            from astropy.io import fits
            meta = fits.getheader(a.target)
        a.telescope, a.target = site_key(meta), meta.get('TARGET')
        rec = get_trigger(a.target, meta=meta)
    else:
        rec = get_trigger(int(a.target), a.telescope)
    if rec is None:
        print("unknown"); sys.exit(1)
    if rec['t0'] is None:
        print("%s %s: no trigger" % (a.telescope, a.target)); sys.exit(1)
    print("%s %s T0=%s (JD %.6f) ra=%.5f dec=%.5f errorbox=%.2f' type=%s %s" % (a.telescope, a.target,
          rec['t0'], rec['jd0'], rec['ra'], rec['dec'], rec['errorbox']*60, rec['type'], rec['name']))
