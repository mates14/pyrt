# Transient search on GRB/EP follow-up sequences: real time and manual reprocessing

`transients.py` (`pyrt-transients`) is the analysis core. It searches a sequence of
calibrated frame catalogues (dophot ECSVs) for two kinds of objects:

- **flare stars** (`pyrt/core/flares.py`): a catalogued star brighter than its catalogue
  magnitude *and* decaying. This is the usual false EP trigger, a red dwarf flare.
- **afterglow candidates** (`pyrt/core/afterglow.py`): a source that is not in the
  catalogue, with a light-curve class and an extrapolation +5/+15/+60 min ahead.

Around it, asarina provides the real-time machinery: the runner `asarina-kinseries`, the
step builders `kinstep` (D50) and `framestep` (SBT), and the reporting tool `trigreport`
(state record, web page, Telegram). The same programs run by hand for reprocessing.

Contents:
1. [How it works](#1-how-it-works)
2. [Real time: D50](#2-real-time-d50)
3. [Real time: SBT](#3-real-time-sbt)
4. [Reports, web page and Telegram](#4-reports-web-page-and-telegram)
5. [Checking the installation without a GRB](#5-checking-the-installation-without-a-grb)
6. [Manual reprocessing](#6-manual-reprocessing)
7. [Reading the output](#7-reading-the-output)
8. [Configuration and files](#8-configuration-and-files)
9. [Known limits](#9-known-limits)

---

## 1. How it works

A follow-up sequence is cut into **steps**. Each step makes about 12 light-curve points
from everything taken so far. That number is enough to see a trend, and constant enough
for the statistics to stay the same from step to step.

| | D50 (Andor C0, kinetic series) | SBT (C1 + C2, short frames) |
|---|---|---|
| unit of data | kinetic series, 96 × 1.25 s (KINFILE dump) | one 5 s (windowed) or 12 s (full) frame |
| step k | the first k series → 12 bins of 8k sub-frames, k ∈ {1, 2, 3, 4, 6, 12} | the first 12k frames → 12 bins of k frames, k = n // 12, k ≤ 12 |
| builder | `asarina.pipeline.kinstep` | `asarina.pipeline.framestep` |
| step 1 is ready | after the first series (~2 min), done ~30 s later | after 12 frames (1–2.5 min), done ~1 min later, search +40 s (full frames, lascaux) |

Each step does the following:

1. **Common grid and deep image.** The step's data go on one pixel grid, and their
   sum is the deep image (12× deeper than a bin).
2. **Deep photometry.** phcat → cat2det → dophot on the deep image, with the spatial
   response held fixed and the zeropoint free. The result is the deep catalogue. Its
   uncatalogued objects are faint afterglow candidates.
3. **Bin photometry.** Forced photometry at the deep positions: `phcat -D deep.fits`,
   which uses SExtractor's dual-image mode for the positions and the numpy aperture
   engine for the magnitudes (see below). Then dophot again, with the deep response
   held fixed and the zeropoint free.
4. **Search.** `pyrt-transients --flare-rec rec.ecsv --deep <deep>.ecsv <bins>` runs
   both searches. It writes `flares.txt`, `afterglow.txt` and `afterglow.ecsv`.
5. **Report.** `trigreport update` builds the state record and the page, publishes
   it, and sends a Telegram message if something new and worth waking up for appeared.

**Photometry engine.** `pyrt-phcat -E numpy` is IRAF's `phot` reimplemented in numpy
(`pyrt/core/aperphot.py`):
- the same `mode` sky, with 3σ rejection about the mode and the upper-middle median;
- the same partial-pixel weighting, clip(ape − r + 0.5, 0, 1);
- the same error formula and output columns.

Compared with IRAF it agrees to 0.0002 mag on bright stars and shows no offset at the
faint end. It is 3–5× faster, and the runner uses it by default. `-E iraf` restores
IRAF, and `-E auto` (phcat's default) picks IRAF when it is installed. SExtractor's own
aperture magnitudes (`-I`) have a sky about 4.6 ADU/px too high on SBT frames, which
makes them 0.05 mag too faint at S/N≈10, so `-I` is not used.

**Trigger information** (time, position, error box) comes from the RTS2 `grb` table via
`pyrt/utils/grbinfo.py`:
- the lookup is a plain read-only select, never `rts2-targetinfo -g`;
- results are cached in `~/.config/pyrt/triggers.dat`;
- triggers younger than 2 days are re-read from the database on every lookup, so a
  refined position (BAT → XRT, a second GBM localisation) is used;
- old ones come from the cache only, and a hand-edited line there wins (the last
  matching line counts).

---

## 2. Real time: D50

**Where:** d (D50 control computer). **Code:** `~/asarina`, `~/pyrt` (editable installs).

1. RTS2 imgproc calls `/etc/rts2/img_process` (as root), which runs
   `asarina-imgproc --realtime`. Its log is `~/imgproc-YYMMDD.log`.
2. imgproc calibrates, solves and photometers the 120 s mean frame (`-RA.fits`), then
   drops root privileges to `mates`.
3. **Step 8.9** applies when `/etc/asarina/config [C0] kinseries_dir` is set and the frame
   is a kinetic series (ACQMODE 3, KINNUM ≥ 12, KINFILE present). `queue_job()` then:
   - writes `~/kinseries/spool/<stem>.json`;
   - starts `asarina-kinseries --dir ~/kinseries` detached, in its own session, so
     imgproc's timeout does not kill it.
4. The runner holds `~/kinseries/lock`, so only one runner works at a time.
   - It groups the jobs into `obs/<date>-<obsid>-<filter>/` and runs the newest step not
     done yet: `asarina.pipeline.kinstep --step k -j <physical cores>`.
   - Then it starts the search and `trigreport update` in the background.
   - It exits after 30 min idle, and the next series starts it again.
5. `trigreport update` uploads the report to lascaux with the restricted key
   `~/.ssh/l50`. That key is `rrsync -wo ~/public_html/`: it can write pages, nothing else.
6. Telegram for D50 is sent by lascaux, from `trigreport scan` in cron (§4).

The trigger lookup is `~/.config/pyrt/trigger-hosts`: `D50 local`, i.e. psql on d itself.

**Timing** (rehearsal, 12 Nov 2026, EP 261106.894): a step is done 25–30 s after its
series arrives. The search adds about 6 s. The flare star was FLARE? at step 1 and
FLARE at step 2.

---

## 3. Real time: SBT

**Where:** lascaux (SBT control computer, which is also the reporting host).
**Code:** `/storage/home/mates/asarina`, `/storage/home/mates/pyrt/main`.

1. RTS2 imgproc calls `/etc/rts2/img_process`, which runs `asarina-imgproc --realtime`
   for every C1/C2 frame. Its log is `~/imgproc-YYMMDD.log`.
2. **Step 8.9** applies when `/etc/asarina/config [C1]` and `[C2]` have
   `kinseries_dir = /home/mates/kinseries` and `kinseries_frames = true`, the target is a
   trigger (TARTYPE G, or GRB_RA in the header) and EXPTIME ≤ `kinseries_max_exptime`
   (15 s). imgproc then copies the calibrated frame into the spool and queues it.
3. The runner groups the frames by observation and filter, with C1 and C2 together.
   - With n frames in, it runs `framestep --step n//12`.
   - framestep resamples every frame onto a common TAN grid, which is the union of the
     step's footprints. C1 and C2 point about 118, −107 px apart, and the bins combine both.
4. Search and report run as on D50. Here the report is copied locally into
   `~/public_html/trigger/sbt-<target>/`, and Telegram is sent at once (`notify`).

The trigger lookup is `SBT local`.

**Which SBT strategies are covered** (`/etc/rts2/scripts/ojitos_adaptive.py`):

| strategy | when | frames | real-time analysis |
|---|---|---|---|
| 1 | box < 15′ | 5 s, 1024² windows, flip/flap | yes |
| 2 | box ≤ 1.5° | 12 s, full frame | yes |
| 3, 4 | box 1.5–4.8° | 60 s, 2×2 / 3×3 mosaic | **no**: one observation covers several fields |
| 5 | trigger > 20 min old | 120 s | **no**: above `kinseries_max_exptime` |

**Timing** (rehearsal on lascaux, 12 Nov 2026: GRB 261024.682, 4096² full frames, numpy
engine, 8 parallel bins):
- step 1 is done 58 s after the 12th frame, step 2 (24 frames) takes 84 s;
- each search takes another ~40 s;
- the runner reports a step's search while the next step is running, so the first
  message does not wait for step 2.

---

## 4. Reports, web page and Telegram

`asarina.pipeline.trigreport`. The page for each trigger is
`https://lascaux.asu.cas.cz/m/trigger/<site>-<target>/`, with site d50, sbt, orm (FRAM-ORM)
or ll (FRAM-Auger). The RTS2 target id is only unique per telescope, hence the prefix.

| file | content |
|---|---|
| `report.json` | the state record, for people and for the observing strategy script: trigger, steps (depth), the best candidates (class, position, mag now, trend, extrapolation, light curve) |
| `lc.png` | light curve of the best candidate, with the bins' limits, the quiescent (catalogue) level and the extrapolation |
| `chart.png` | the deep image around the best candidate or the trigger position, with the error box |
| `gcn.txt` | a draft GCN circular, **never sent** |
| `index.html` | all of it, reloading itself |

**Telegram.** The bot and chat come from lascaux `~/gcn/config.json` (`telegram.bot_token`,
and the chat under the key `private_chat_id` by default; `--chat chat_id` for the team).
The token never leaves lascaux.

A message goes out only when the best candidate is new or changed its class, and only
for:
- an ordered afterglow: FADING, RISING or TURNING;
- a decaying flare: FLARE;
- a flare-star candidate inside the error box: FLARE?.

A steady NEW, an ERRATIC or a SPIKE stays on the page. `notified.json` in the page
directory remembers what was sent.

**Wide (GBM) boxes.** When the box radius is over 30′, only a source that varied while we
watched (FADING, RISING, TURNING, FLARE) becomes the best candidate or a message. A
catalogued star that is ≥ 0.2 mag (≥ 5σ) brighter than the catalogue but steady gets one
sentence in the draft GCN ("…constant during our observation; we do not consider it
related"), and nothing else.

**Notifying D50 triggers.** The D50 uploads cannot run anything on lascaux, so a cron job
there notifies for pages updated since the last scan:

```
* * * * * cd /tmp && /usr/bin/python3 -m asarina.pipeline.trigreport scan ~/public_html/trigger >> ~/trigreport-scan.log 2>&1
```

**This line must be in mates' crontab on lascaux**, or D50 triggers get pages but no
messages. (It was not installed as of 12 Nov 2026.)

---

## 5. Checking the installation without a GRB

The real-time chain can be rehearsed with archived data through the real entry point:
`queue_job()`, exactly as imgproc calls it. It starts the runner detached and publishes
the report, so the only differences from production are the report root and the chat.

On d (EP 261106.894, three kinetic series at their real spacing):

```
cd /tmp && env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME=/home/mates USER=mates LOGNAME=mates TERM=xterm \
  TRIGREPORT_ROOT="~/public_html/trigger-test" TRIGREPORT_URL=https://lascaux.asu.cas.cz/m/trigger-test \
  TRIGREPORT_TELEGRAM_CONFIG=/nonexistent \
  /usr/bin/python3 ~/kinseries-test/rehearse.py /home/mates/kinseries-rehearsal 3 110
tail -f ~/kinseries-rehearsal/kinseries.log
```

On lascaux (GRB 261024.682, 29 C1/C2 frames, spacing compressed to 4 s), the same
environment and:

```
/usr/bin/python3 ~/kinseries-test/rehearse_sbt.py /home/mates/kinseries-rehearsal 4
```

- `TRIGREPORT_TELEGRAM_CONFIG=/nonexistent` blocks the message.
- `python3 -m asarina.pipeline.trigreport notify <pagedir> --dry-run` prints what would
  have been sent.
- Afterwards kill the rehearsal runner (it would exit after 30 min idle anyway):
  `pkill -f 'kinseries --dir /home/mates/kinseries-rehearsal'`.

**What a rehearsal does not cover:**
- imgproc itself on a fresh frame (calibration, astrometry, the queueing condition on a
  live header);
- RTS2 invoking imgproc;
- the live grb row.

A faked trigger covers all of those. Faking one (a G-type target with a grb row) also
needs a real Telegram message for a full check. Point it at the private chat
(`TRIGREPORT_TELEGRAM_CHAT=private_chat_id`, the default).

**Things to look at after a real trigger:**
- `~/kinseries/kinseries.log`: steps, timing, the candidates;
- `~/kinseries/obs/*/stepNN/{kinstep,framestep}.log`, `transients.log`;
- `~/kinseries/obs/*/trigreport.log`: upload and notify errors;
- `~/imgproc-YYMMDD.log`: "Kinetic series queued" lines, or `kinseries:` warnings;
- on lascaux, `~/trigreport-scan.log`.

---

## 6. Manual reprocessing

Everything below runs as an ordinary user on any machine with pyrt and asarina. Use
`--engine iraf` to compare with IRAF.

### D50 kinetic series

```
# step k from the first k series (raw RA.fits with their KINFILE dumps, or *-kin.fits cubes)
python3 -m asarina.pipeline.kinstep --step 2 -j 4 -o out/step02 S1-RA.fits S2-RA.fits ...
```

The reference ECSV of each series (its 120 s mean, solved by the pipeline) is found by
the frame stem: next to the input, in its parent directory, or in `~/phdb-rt`. `--ref`
gives it explicitly. Dark and flat come from the raw headers' calibration, or from
`--dark`/`--flat`.

### SBT (or any) short frames

```
python3 -m asarina.pipeline.framestep --step 1 -j 8 \
    --filter-terms "#F1=0.540,#F3=0.165,#F4=0.024,#PAC=0.0446" \
    -o out/step01 f1-df.fits:f1-df.ecsv f2-df.fits:f2-df.ecsv ...
```

- **Frames:** use the frames of one filter only, in time order. The runner never mixes
  filters.
- **Colour terms:** the `--filter-terms` above are the production terms for SBT clear
  (`/etc/asarina/config` on lascaux). Without the option, framestep reads them from
  the local config.
- **Step k:** `--step k` uses the first 12k frames.

### The search

In the step's output directory:

```
pyrt-transients --flare-rec rec.ecsv --deep <stem>-deep<k>.ecsv *-[kf][0-9]*b*.ecsv > transients.log
```

- `--no-flares` / `--no-afterglow` skip one of the searches.
- `--flare-window M` uses only the first M minutes.
- `-c` overrides the catalogue.
- The trigger is looked up by the frames' TARGET and TELESCOP (§1). For old data,
  add a line to `~/.config/pyrt/triggers.dat`:

```
# telescope target t0(UTC ISO) ra[deg] dec[deg] errorbox[deg] type name
SBT 64517 2026-10-24T16:23:00.790 320.330000 61.810000 1.41 115 GRB 261024.682
```

### The report

```
TRIGREPORT_ROOT='~/public_html/trigger-test' TRIGREPORT_URL='https://lascaux.asu.cas.cz/m/trigger-test' \
  python3 -m asarina.pipeline.trigreport update <obsdir> --dry-run
```

- `<obsdir>` holds the `stepNN/` directories (the newest step is reported).
- `--dry-run` prints the Telegram text instead of sending it.
- Keep test pages under `trigger-test/`.
- `build` only builds; `publish` uploads an already-built report.

### A whole replay

Run the steps one after another as the runner would (1, 2, 3, 4, 6, 12 for D50; 1, 2,
… for frames), each followed by the search in its directory and the report on the
parent directory. On d, `~/eptest.sh TARGET NIGHT_DIR` does that for a D50 EP night.
For the full real-time path with archived data, see §5.

---

## 7. Reading the output

**`flares.txt`**, one line per candidate:
```
FLARE? 46.94707 +24.96443 mag 13.23 excess +0.198±0.060 (3.3σ ...) fading ... decay 1.0σ tau 11 min,
       flare +0.17 now, +0.01 in 30 min S=3.5+2.5 | colours ... pm 262 mas/y H=16.4 | 1.5' = 0.48 box [red,dwarf,inbox] n=12/1.6 min
```

- **FLARE** means brighter than the catalogue *and* decaying (S ≥ 5).
- **FLARE?** means significantly brighter, but the decay is not established yet.
- **S** is the combined significance, + the priors (in the box, red, dwarf by
  reduced proper motion).
- A FLARE? whose catalogue magnitude is inconsistent with the neighbouring bands
  (colours of opposite sign, both > 0.3 mag) is dropped as a catalogue error.

**`afterglow.txt`**, one line per candidate:
```
NEW 320.38586 +62.60027 mag 12.63 (last 12.59, 2.6 above limit) var -1.3σ cont 0.93 alpha -0.02±0.10
    -> +5' 12.62±0.16 +15' 12.61±0.37 +60' 12.61±0.85 S=7.1 | 47.4' = 0.56 box n=12/7.6 min
```

| class | meaning |
|---|---|
| NEW | steady within the errors |
| FADING / RISING | significant, ordered variation (continuity < 0.6), the trend's sign |
| TURNING | ordered and curved: a rise into a decay or the reverse |
| ERRATIC | significant but not ordered: a bogus, or a flaring source (a GRB may flare) |
| SPIKE | all variation in one interior bin: a mover or a cosmic, kept only inside the box |

- **alpha** is the power-law index in time since T0 (> 0 fading).
- **The extrapolations** use the recent half of the curve in log time. Plan on their
  faint end: decays often steepen.

---

## 8. Configuration and files

| what | where |
|---|---|
| real-time switch, D50 | d `/etc/asarina/config [C0] kinseries_dir` |
| real-time switch, SBT | lascaux `/etc/asarina/config [C1],[C2] kinseries_dir, kinseries_frames = true, kinseries_max_exptime` |
| colour terms (dophot) | the same files, per filter |
| trigger database hosts | `~/.config/pyrt/trigger-hosts` (`D50 local` on d, `SBT local` on lascaux) |
| trigger cache | `~/.config/pyrt/triggers.dat` |
| runner directory | `~/kinseries/` (spool, obs, kinseries.log, lock) |
| upload key (d → lascaux) | d `~/.ssh/l50`, lascaux `authorized_keys`: `command="rrsync -wo /home/mates/public_html/",restrict` |
| Telegram bot | lascaux `~/gcn/config.json` |
| pages | lascaux `~/public_html/trigger/` (tests: `trigger-test/`) |

**Environment overrides** (for tests):

| variable | effect |
|---|---|
| `TRIGREPORT_ROOT` | where the pages go |
| `TRIGREPORT_URL` | the page URL in messages |
| `TRIGREPORT_HOST` | the reporting host |
| `TRIGREPORT_SSH_KEY` | the upload key |
| `TRIGREPORT_TELEGRAM_CONFIG` | the bot and chat config file |
| `TRIGREPORT_TELEGRAM_CHAT` | which chat key in that file |
| `PYRT_TRIGGERS` | the trigger cache file |
| `PYRT_TRIGGER_HOSTS` | the trigger database hosts file |

**Runner options** (`asarina-kinseries`):
- `-j` parallel bins, default the physical cores (8 on lascaux, 4 on d);
- `--idle 1800`;
- `--timeout 900` per step;
- `--kinstep-args '...'`;
- `--once`.

**Code updates.** The runner is started fresh for every sequence, so a `git pull` of
pyrt/asarina takes effect at the next trigger. imgproc is a new process per frame.

---

## 9. Known limits

- **SBT mosaics and late follow-ups** are not analysed in real time (§3).
- **Mixed filters** (seen on 24 Oct 2026: C2 still in r after interrupted flats) form
  separate chains. The r chain then usually has too few frames for a step.
- **Wide boxes:** a constant brightened star is never an alert, by design (GBM policy, §4).
- **Comets and slow movers:** extended deep objects (> 1.5× FWHM) are dropped, and
  steady NEW sources do not alert. A cometary class is planned separately.
- **Memory on lascaux:** framestep needs ~3 GB plus ~0.5 GB per parallel bin, about
  7 GB with 8 bins. Watch it if lascaux is busy.
- **The trigger lookup needs the database:** if it is unreachable, the report has no
  trigger information (no T0, no box). Add a line to `triggers.dat` by hand.
