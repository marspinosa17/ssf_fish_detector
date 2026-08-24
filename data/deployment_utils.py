"""
Timestamp parsing and deployment grouping for PAM fish detection audio files.

Filename conventions differ by source and, within Xavier, by sub-collection.
Patterns catalogued by sampling every audio file under XavierData, SethData,
and TagusData (see the inventory report printed by running this module
directly):

  xavier_amar_dotted      AMAR173.4.20190915T184248Z.wav
                          {device}.{chan}.{YYYYMMDDTHHMMSS}Z.ext
  xavier_jasco_underscore JASCOAMARHYDROPHONE742_20140913T032516.038Z.wav
                          {device}_{YYYYMMDDTHHMMSS}.{ms}Z.ext
  xavier_dfo_device_dot   67674121.181018053806.wav
                          {device}.{YYMMDDHHMMSS}.ext  (2-digit year)
  seth_site_underscore    CL_170411040502.wav
                          {site}_{YYMMDDHHMMSS}.wav
  tagus_datetime_plain    20170116_1130_.wav
                          {YYYYMMDD}_{HHMM}_.wav
  tagus_datetime_offset   20161117_0757_-08.000-08.167.wav
                          {YYYYMMDD}_{HHMM}_ + duty-cycle offset suffix
  tagus_montijo           Montijo_20210712_182600.wav
                          Montijo_{YYYYMMDD}_{HMS, variable width}.wav

Every file sampled across both datasets matched one of these seven patterns
(0 unmatched) — see build the report with `python deployment_utils.py`.

Deployment definition per source (see report/PR description for the full
rationale):
  xavier: parent folder. Each Dataset_1/2/3 sub-collection folder was
          checked and contains recordings from a single physical device at
          a single site, so the folder already is the deployment.
  seth:   only 3 site folders (CL/TK/YA) exist, but each spans ~7 weeks of
          near-continuous 30-min-cadence recording with a handful of
          multi-day gaps — folder-level grouping is too coarse (only 3
          groups) and per-file grouping is too fine (throws away session
          structure and leaks near-duplicate audio across splits). A
          deployment is instead a "session": a run of recordings at one
          site with no gap larger than SETH_SESSION_GAP_S between
          consecutive files. Real gaps in the data cluster below 1hr
          (normal cadence) and above ~24hr (missed days / redeployment),
          so any threshold in that range gives the same grouping; 6h is
          used for margin.
  tagus:  the existing train/validation/m_knocks folder split does NOT
          align with deployments — the same calendar date (e.g.
          2017-01-16) appears split across both train/ and validation/,
          so trusting the folder would put recordings from one field
          session on both sides of a split. Each field day is a handful of
          short duty-cycled recordings (e.g. dawn/midday/dusk), so a
          deployment is the calendar date parsed from the filename,
          independent of which folder the file lives in.
  hawaii: filenames (e.g. OLOW_SEP14005_0017.wav) carry no embedded
          timestamp, only a device/session tag and a running file index —
          {tag}_{index}.ext. A deployment is the device/session tag
          (everything before the final underscore-index), which is also
          all that split assignment needs: hawaii is never divided into
          train/val/test (see build_manifest.py), so deployment_id here is
          just for reporting, not split logic.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# How large a gap (seconds) between consecutive Seth recordings at the same
# site must be before it's treated as a new deployment session rather than a
# missed/duty-cycled file. Observed gaps cluster at <=3600s (normal cadence)
# or >=87000s (~24h, real interruptions) with nothing in between, so this is
# not a sensitive choice.
SETH_SESSION_GAP_S = 6 * 3600

# ============================================================================
# FILENAME PATTERNS
# ============================================================================

_XAVIER_AMAR = re.compile(r'^([A-Za-z]+\d+)\.(\d+)\.(\d{8}T\d{6})Z\.\w+$')
_XAVIER_JASCO = re.compile(r'^([A-Za-z0-9]+)_(\d{8}T\d{6}(?:\.\d+)?)Z\.\w+$')
_XAVIER_DFO = re.compile(r'^(\d+)\.(\d{12})\.\w+$')
_SETH_SITE = re.compile(r'^([A-Za-z]+)_(\d{12})\.\w+$')
_TAGUS_MONTIJO = re.compile(r'^Montijo_(\d{8})_(\d{1,6})\.\w+$')
_TAGUS_OFFSET = re.compile(r'^(\d{8})_(\d{4})_-\d+\.\d+--?\d+\.\d+\.\w+$')
_TAGUS_PLAIN = re.compile(r'^(\d{8})_(\d{4})_\.\w+$')
_HAWAII_OLOWALU = re.compile(r'^(.+)_(\d+)\.\w+$')


def _parse_xavier(name: str) -> Optional[datetime]:
    m = _XAVIER_AMAR.match(name)
    if m:
        _, _, ts = m.groups()
        return datetime.strptime(ts, '%Y%m%dT%H%M%S')

    m = _XAVIER_JASCO.match(name)
    if m:
        _, ts = m.groups()
        fmt = '%Y%m%dT%H%M%S.%f' if '.' in ts else '%Y%m%dT%H%M%S'
        return datetime.strptime(ts, fmt)

    m = _XAVIER_DFO.match(name)
    if m:
        _, ts = m.groups()
        return datetime.strptime(ts, '%y%m%d%H%M%S')

    return None


def _parse_seth(name: str) -> Optional[datetime]:
    m = _SETH_SITE.match(name)
    if m:
        _, ts = m.groups()
        return datetime.strptime(ts, '%y%m%d%H%M%S')
    return None


def _parse_tagus(name: str) -> Optional[datetime]:
    m = _TAGUS_MONTIJO.match(name)
    if m:
        date, hms = m.groups()
        return datetime.strptime(date + hms.zfill(6), '%Y%m%d%H%M%S')

    m = _TAGUS_OFFSET.match(name)
    if m:
        date, hm = m.groups()
        return datetime.strptime(date + hm, '%Y%m%d%H%M')

    m = _TAGUS_PLAIN.match(name)
    if m:
        date, hm = m.groups()
        return datetime.strptime(date + hm, '%Y%m%d%H%M')

    return None


_PARSERS = {
    'xavier': _parse_xavier,
    'seth':   _parse_seth,
    'tagus':  _parse_tagus,
}


def parse_timestamp(audio_path, source: str) -> Optional[datetime]:
    """Extract the recording start timestamp from an audio filename.

    Returns None (without raising) if `source` is unrecognized or the
    filename doesn't match any known pattern for that source. Callers that
    need to know about parse failures should check for None explicitly —
    this function does not log, so bulk validation should be done via
    `build_inventory_report` instead of relying on scattered warnings.
    """
    parser = _PARSERS.get(source)
    if parser is None:
        return None
    return parser(Path(audio_path).name)


# ============================================================================
# DEPLOYMENT GROUPING
# ============================================================================

# Cache of {wavs_dir: {filename: session_index}} for Seth, built lazily
# per-directory the first time a file from that directory is seen.
_seth_session_cache: Dict[Path, Dict[str, int]] = {}


def _get_seth_session_map(wavs_dir: Path) -> Dict[str, int]:
    if wavs_dir in _seth_session_cache:
        return _seth_session_cache[wavs_dir]

    entries = []
    for f in wavs_dir.glob('*.wav'):
        ts = _parse_seth(f.name)
        if ts is not None:
            entries.append((ts, f.name))
    entries.sort(key=lambda e: e[0])

    session_map = {}
    session_id = 0
    prev_ts = None
    gap_threshold = timedelta(seconds=SETH_SESSION_GAP_S)
    for ts, name in entries:
        if prev_ts is not None and (ts - prev_ts) > gap_threshold:
            session_id += 1
        session_map[name] = session_id
        prev_ts = ts

    _seth_session_cache[wavs_dir] = session_map
    return session_map


def get_deployment_id(audio_path, source: str) -> str:
    """Group an audio file into the deployment it belongs to.

    See module docstring for the reasoning behind each source's definition.
    Falls back to the parent-directory path (with a logged warning) if the
    filename can't be parsed, so every file still gets *some* group.
    """
    audio_path = Path(audio_path)

    if source == 'xavier':
        return str(audio_path.parent)

    if source == 'seth':
        site = audio_path.parent.parent.name
        session_map = _get_seth_session_map(audio_path.parent)
        session = session_map.get(audio_path.name)
        if session is None:
            logger.warning(f"Seth: could not parse timestamp for "
                            f"{audio_path.name}, falling back to parent dir")
            return str(audio_path.parent)
        return f"{site}_session{session:02d}"

    if source == 'tagus':
        ts = _parse_tagus(audio_path.name)
        if ts is None:
            logger.warning(f"Tagus: could not parse timestamp for "
                            f"{audio_path.name}, falling back to parent dir")
            return str(audio_path.parent)
        return f"tagus_{ts.strftime('%Y%m%d')}"

    if source == 'hawaii':
        m = _HAWAII_OLOWALU.match(audio_path.name)
        if m is None:
            logger.warning(f"Hawaii: could not parse device tag for "
                            f"{audio_path.name}, falling back to parent dir")
            return str(audio_path.parent)
        return m.group(1)

    return str(audio_path.parent)


# ============================================================================
# INVENTORY REPORT
# ============================================================================

def build_inventory_report(xavier_root: Path, seth_root: Path, tagus_root: Path) -> Dict:
    """Scan every audio file under each root and report parse/deployment stats."""
    roots = {'xavier': xavier_root, 'seth': seth_root, 'tagus': tagus_root}
    report = {}

    for source, root in roots.items():
        files = [f for f in root.rglob('*') if f.suffix.lower() in ('.wav', '.flac')]
        parsed, unparsed = [], []
        for f in files:
            ts = parse_timestamp(f, source)
            (parsed if ts is not None else unparsed).append(f)

        deployments = {get_deployment_id(f, source) for f in files}

        report[source] = {
            'total_files': len(files),
            'parsed': len(parsed),
            'unparsed': len(unparsed),
            'unparsed_paths': unparsed,
            'n_deployments': len(deployments),
        }

    return report


def print_inventory_report(report: Dict) -> None:
    print("=" * 80)
    print("DEPLOYMENT / TIMESTAMP INVENTORY REPORT")
    print("=" * 80)
    for source, stats in report.items():
        print(f"\n{source}:")
        print(f"  total audio files : {stats['total_files']}")
        print(f"  parsed timestamps : {stats['parsed']}")
        print(f"  unparsed          : {stats['unparsed']}")
        for p in stats['unparsed_paths'][:20]:
            print(f"    ? {p}")
        print(f"  distinct deployments: {stats['n_deployments']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Reuse the same path config as build_manifest.py
    from build_manifest import XAVIER_DATA_PATH, SETH_DATA_PATH, TAGUS_DATA_PATH

    report = build_inventory_report(XAVIER_DATA_PATH, SETH_DATA_PATH, TAGUS_DATA_PATH)
    print_inventory_report(report)
