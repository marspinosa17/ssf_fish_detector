"""
Build a unified dataset manifest for PAM fish detection project.

Parses annotations from three sources:
- XavierData: Raven Pro TXT (tab-separated), binary fish/no-fish labels, multiple sample rates
- SethData: CSV with time/frequency bounds and call types, 48 kHz WAV
- TagusData: Raven Pro TXT, species labels, 4 kHz WAV, pre-split into train/val

Outputs: manifest.csv with columns:
    clip_id, source, audio_path, begin_s, end_s, duration_s,
    low_hz, high_hz, label, is_fish, use_for_detection, split,
    deployment_id, timestamp
"""

import pandas as pd
import numpy as np
import soundfile as sf
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x

from deployment_utils import parse_timestamp, get_deployment_id

# ============================================================================
# CONFIG
# ============================================================================

DATA_ROOT        = Path(os.environ.get("FD_DATA_ROOT", Path(__file__).resolve().parent))
XAVIER_DATA_PATH = DATA_ROOT / "XavierData"
SETH_DATA_PATH   = DATA_ROOT / "SethData"
TAGUS_DATA_PATH  = DATA_ROOT / "TagusData"
HAWAII_DATA_PATH = DATA_ROOT / "hawaii"
OUTPUT_PATH      = DATA_ROOT / "manifest.csv"

TRAIN_RATIO = 0.70  # implied as the remainder after TEST_RATIO and VAL_RATIO
VAL_RATIO   = 0.15  # fraction of EACH non-test deployment's own time, held out as "the future"
TEST_RATIO  = 0.15  # fraction of EACH source's total deployment duration, held out entirely

RANDOM_SEED = 42
MIN_FILES_FOR_VAL_SPLIT = 3  # deployments with fewer files go entirely to train

# Hawaii (Olowalu) is a small, separately-collected subset kept out of the
# train/val/test rotation entirely -- every hawaii row gets split='hawaii'
# directly (see file_split population in main()), never touched by the
# deployment-duration split logic above. Used only as a standalone eval set.
# SOURCE_EXPECTED_SPLITS drives the split guardrail at the end of main():
# each source's rows must cover exactly these splits, non-empty, or the run
# fails loudly instead of silently shipping a source with a missing split.
SOURCE_EXPECTED_SPLITS = {
    'xavier': ('train', 'val', 'test'),
    'seth':   ('train', 'val', 'test'),
    'tagus':  ('train', 'val', 'test'),
    'hawaii': ('hawaii',),
}

# Tagus deployments are 1-6 short files each (vs. hundreds for Xavier/Seth),
# so the chronological 85/15-by-duration cut used for those two sources is
# meaningless at that scale — e.g. a 3-file deployment's closest-to-85%
# split point is "keep all 3 in train", so it would silently never produce
# val data. Instead, Tagus's TEST deployments are chosen with the same
# duration-based selection as Xavier/Seth (this is what actually fixes the
# "zero test files" gap), while train/val within the surviving deployments
# trusts the train/validation folder split the dataset already ships with.
# Set to False to leave Tagus with no test data at all (today's behavior).
FOLD_TAGUS_TEST_INTO_DEPLOYMENT_SPLIT = True

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# HELPERS
# ============================================================================

def _clean_float(val, default=None):
    """Strip leading apostrophes/whitespace and convert to float.

    If the cleaned string is empty or 'nan', returns default when provided,
    otherwise raises ValueError (causing the row to be silently skipped).
    """
    s = str(val).lstrip("'").strip()
    if s in ('', 'nan'):
        if default is not None:
            return default
        raise ValueError(f"Cannot convert {val!r} to float")
    return float(s)


def build_audio_index(root_path: Path) -> Dict[str, List[Path]]:
    """Pre-build an index of audio files keyed by stem for fast lookup."""
    index = defaultdict(list)
    logger.info(f"Building audio index for {root_path.name}...")
    for audio_file in root_path.rglob('*'):
        if audio_file.suffix.lower() in ('.wav', '.flac'):
            index[audio_file.stem].append(audio_file)
    logger.info(f"  Found {len(index)} unique audio files")
    return dict(index)


def find_audio_file(annotation_path: Path, filename_hint: Optional[str] = None,
                    audio_index: Optional[Dict] = None) -> Optional[Path]:
    """Resolve an annotation file to its corresponding audio file."""
    annotation_dir = annotation_path.parent
    base_name = annotation_path.stem.rsplit('.Table', 1)[0].replace('.chan1', '')

    if filename_hint and filename_hint != 'nan':
        hint_base = Path(filename_hint).stem
        if audio_index and hint_base in audio_index:
            for cand in audio_index[hint_base]:
                if cand.parent in (annotation_dir, annotation_dir.parent):
                    return cand
            return audio_index[hint_base][0]

    if audio_index and base_name in audio_index:
        for cand in audio_index[base_name]:
            if cand.parent in (annotation_dir, annotation_dir.parent):
                return cand
        return audio_index[base_name][0]

    for ext in ('.wav', '.WAV', '.flac', '.FLAC'):
        matches = list(annotation_dir.glob(f'*{ext}'))
        if matches:
            return matches[0]

    return None


def get_audio_durations(audio_paths: List[str]) -> Dict[str, float]:
    """Read the real duration (seconds) of each unique audio file via soundfile."""
    durations = {}
    for p in tqdm(audio_paths, desc="Reading durations", disable=not HAS_TQDM):
        try:
            info = sf.info(p)
            durations[p] = info.frames / info.samplerate
        except Exception as e:
            logger.debug(f"Could not read duration for {p}: {e}")
    return durations


def get_binary_fish_label(label: str, source: str) -> Optional[bool]:
    """Convert a source-specific label to a binary fish flag.

    Xavier: FS=True, NN/HS/KW=False, UN=None (uncertain), others=True
    Seth:   pulse/tonal/knock/grunt/growl/click=True,
            chorus=None (ignored — poor bounding-box fit / low detection recall,
            see manifest doc), boat/ship/background/noise=False, others=True
    Tagus:  all species (m, w, lt) = True
    Hawaii: pulse/tonal=True (same fish-call vocabulary as Seth),
            chorus=None (same rationale as Seth), unknown=None (uncertain,
            mirrors Xavier's UN), whale?=None (the '?' marks annotator
            uncertainty, so treated the same as the other uncertain labels
            rather than guessed as non-fish), others=True
    """
    label_lower = str(label).strip().lower()

    if source == 'xavier':
        if label_lower == 'fs':
            return True
        if label_lower in ('nn', 'hs', 'kw'):
            return False
        if label_lower == 'un':
            return None
        return True

    if source == 'seth':
        fish_calls = {'pulse', 'tonal', 'knock', 'grunt', 'growl', 'click'}
        non_fish   = {'boat', 'ship', 'background', 'noise'}
        if label_lower == 'chorus':
            return None
        if label_lower in fish_calls:
            return True
        if label_lower in non_fish:
            return False
        return True

    if source == 'tagus':
        return True

    if source == 'hawaii':
        uncertain = {'chorus', 'unknown', 'whale?'}
        if label_lower in uncertain:
            return None
        return True

    return None


def select_test_deployments(deployment_durations: Dict[str, float], target_frac: float,
                             seed: int) -> set:
    """Seeded-shuffle + greedy duration accumulation to choose held-out test deployments.

    Deployments are visited in random (seeded) order and added to the test
    set as long as doing so keeps the running total under target_frac of
    the source's total duration. At the boundary, whichever of
    "include" / "exclude" the last-considered deployment lands closer to
    the target is kept — a largest-remainder-style correction, applied
    here to a stop condition instead of a proportional split across bins.
    """
    names = list(deployment_durations.keys())
    total = sum(deployment_durations.values())
    if total <= 0 or not names:
        return set()
    target = total * target_frac

    rng = np.random.RandomState(seed)
    shuffled = [names[i] for i in rng.permutation(len(names))]

    selected, running = [], 0.0
    for name in shuffled:
        if running >= target:
            break
        selected.append(name)
        running += deployment_durations[name]

    if selected:
        running_without_last = running - deployment_durations[selected[-1]]
        if abs(running_without_last - target) < abs(running - target):
            selected.pop()

    return set(selected)


def assign_deployment_train_val(files: pd.DataFrame, val_frac: float) -> Dict[str, str]:
    """Chronologically split one deployment's files into train/val by duration.

    Sorts by timestamp and walks in order, keeping files in train while the
    running duration total stays under (1 - val_frac) of the deployment's
    total; the rest (the deployment's own most recent recordings) go to
    val. Whole files are assigned to one side — a file is never split.
    """
    files = files.sort_values('timestamp')
    total_dur = files['duration'].sum()
    target_train = total_dur * (1 - val_frac)

    assignment = {}
    running = 0.0
    for row in files.itertuples(index=False):
        assignment[row.audio_path] = 'train' if running < target_train else 'val'
        running += row.duration
    return assignment


# ============================================================================
# PARSERS
# ============================================================================

def parse_raven_txt(annotation_path: Path, source_label: str,
                    audio_index: Optional[Dict] = None) -> List[Dict]:
    """Parse a Raven Pro TXT annotation file."""
    records = []
    try:
        df = pd.read_csv(annotation_path, sep='\t', dtype=str, low_memory=False)
        df.columns = [col.strip().lower() for col in df.columns]

        for idx, row in df.iterrows():
            try:
                begin_s = _clean_float(row.get('begin time (s)', 0), default=0.0)
                end_s   = _clean_float(row.get('end time (s)',   0), default=0.0)
                low_hz  = _clean_float(row.get('low freq (hz)',  0), default=0.0)
                high_hz = _clean_float(row.get('high freq (hz)', 1400), default=1400.0)

                label = None
                for col in df.columns:
                    if 'category' in col or 'class' in col:
                        label = str(row[col]).strip()
                        break
                if not label or label == 'nan':
                    label = 'fish' if source_label == 'xavier' else 'unknown'

                audio_path = None
                for col in df.columns:
                    if 'file' in col and 'path' not in col and 'offset' not in col:
                        hint = str(row[col]).strip()
                        if hint and hint != 'nan':
                            audio_path = find_audio_file(annotation_path, hint, audio_index)
                            break
                if not audio_path:
                    audio_path = find_audio_file(annotation_path, None, audio_index)

                if audio_path:
                    records.append({
                        'clip_id':    f"{audio_path.stem}_{idx}",
                        'source':     source_label.lower(),
                        'audio_path': str(audio_path.absolute()),
                        'begin_s':    begin_s,
                        'end_s':      end_s,
                        'low_hz':     low_hz,
                        'high_hz':    high_hz,
                        'label':      label,
                        'is_fish':    get_binary_fish_label(label, source_label.lower()),
                        'split':      None,
                    })
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Error parsing {annotation_path}: {e}")

    return records


def parse_seth_csv(annotation_path: Path, audio_index: Optional[Dict] = None) -> List[Dict]:
    """Parse a SethData CSV annotation file."""
    records = []
    try:
        df = pd.read_csv(annotation_path)
        audio_path = find_audio_file(annotation_path, None, audio_index)
        if not audio_path:
            return records

        for idx, row in df.iterrows():
            try:
                label = str(row['label']).strip()
                records.append({
                    'clip_id':    f"{audio_path.stem}_{idx}",
                    'source':     'seth',
                    'audio_path': str(audio_path.absolute()),
                    'begin_s':    _clean_float(row['t_start']),
                    'end_s':      _clean_float(row['t_stop']),
                    'low_hz':     _clean_float(row['f_min']),
                    'high_hz':    _clean_float(row['f_max']),
                    'label':      label,
                    'is_fish':    get_binary_fish_label(label, 'seth'),
                    'split':      None,
                })
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Error parsing {annotation_path}: {e}")

    return records


# ============================================================================
# SOURCE PARSING
# ============================================================================

def filter_xavier_annotation_files(txt_files: List[Path]) -> List[Path]:
    """For Dataset_1, keep only the aggregate annotation file per sublocation.

    Dataset_1_Strait_of_Georgia_BC ships one aggregate annotation file per
    sublocation (Annotations_dataset_*.Table.1.selections.txt) that already
    covers every individual recording. Keeping only these prevents double-counting.
    """
    dataset1 = [f for f in txt_files if 'Dataset_1_Strait_of_Georgia' in str(f)]
    others   = [f for f in txt_files if 'Dataset_1_Strait_of_Georgia' not in str(f)]
    main     = [f for f in dataset1  if 'Annotations_dataset_' in f.name]
    logger.info(f"  Dataset_1: keeping {len(main)} aggregate files "
                f"(filtered from {len(dataset1)} total)")
    return main + others


def parse_xavier_data(audio_index: Dict) -> List[Dict]:
    """Parse all XavierData Raven Pro TXT annotation files."""
    logger.info("Parsing XavierData...")
    txt_files = [f for f in XAVIER_DATA_PATH.rglob("*.txt") if f.name != "ReadMe.txt"]
    txt_files = filter_xavier_annotation_files(txt_files)
    logger.info(f"  Processing {len(txt_files)} annotation files")
    records = []
    for f in tqdm(txt_files, desc="XavierData", disable=not HAS_TQDM):
        records.extend(parse_raven_txt(f, "xavier", audio_index))
    logger.info(f"  Parsed {len(records)} records")
    return records


def parse_seth_data(audio_index: Dict) -> List[Dict]:
    """Parse all SethData CSV annotation files."""
    logger.info("Parsing SethData...")
    csv_files = list(SETH_DATA_PATH.rglob("*.csv"))
    logger.info(f"  Found {len(csv_files)} annotation files")
    records = []
    for f in tqdm(csv_files, desc="SethData", disable=not HAS_TQDM):
        records.extend(parse_seth_csv(f, audio_index))
    logger.info(f"  Parsed {len(records)} records")
    return records


def parse_tagus_data(audio_index: Dict) -> List[Dict]:
    """Parse all TagusData Raven Pro TXT files."""
    logger.info("Parsing TagusData...")
    txt_files = list(TAGUS_DATA_PATH.rglob("*.txt"))
    logger.info(f"  Found {len(txt_files)} annotation files")
    records = []
    for f in tqdm(txt_files, desc="TagusData", disable=not HAS_TQDM):
        recs  = parse_raven_txt(f, "tagus", audio_index)
        records.extend(recs)
    logger.info(f"  Parsed {len(records)} records")
    return records


def parse_hawaii_txt(annotation_path: Path, audio_index: Dict) -> List[Dict]:
    """Parse one Hawaii (Olowalu) label TXT: tab-separated, no header,
    columns begin_s, end_s, low_hz, high_hz, label (same shape as SethData's
    CSV, just tab-delimited and headerless).

    The audio file is the same stem with the '_labels' suffix removed:
    wav_labels/OLOW_SEP14005_0017_labels.txt -> wavs/OLOW_SEP14005_0017.wav.
    """
    records = []
    stem = annotation_path.stem
    audio_stem = stem[:-len('_labels')] if stem.endswith('_labels') else stem
    candidates = audio_index.get(audio_stem)
    if not candidates:
        logger.debug(f"Hawaii: no audio match for {annotation_path.name}")
        return records
    audio_path = candidates[0]

    try:
        df = pd.read_csv(annotation_path, sep='\t', header=None,
                         names=['t_start', 't_stop', 'f_min', 'f_max', 'label'],
                         dtype=str)
    except Exception as e:
        logger.debug(f"Error parsing {annotation_path}: {e}")
        return records

    for idx, row in df.iterrows():
        try:
            label = str(row['label']).strip()
            records.append({
                'clip_id':    f"{audio_path.stem}_{idx}",
                'source':     'hawaii',
                'audio_path': str(audio_path.absolute()),
                'begin_s':    _clean_float(row['t_start']),
                'end_s':      _clean_float(row['t_stop']),
                'low_hz':     _clean_float(row['f_min']),
                'high_hz':    _clean_float(row['f_max']),
                'label':      label,
                'is_fish':    get_binary_fish_label(label, 'hawaii'),
                'split':      None,
            })
        except Exception:
            pass

    return records


def parse_hawaii_data(audio_index: Dict) -> List[Dict]:
    """Parse all Hawaii (Olowalu) label TXT files under hawaii/wav_labels."""
    logger.info("Parsing HawaiiData...")
    label_dir = HAWAII_DATA_PATH / "wav_labels"
    txt_files = sorted(label_dir.glob("*_labels.txt"))
    logger.info(f"  Found {len(txt_files)} annotation files")
    records = []
    for f in tqdm(txt_files, desc="HawaiiData", disable=not HAS_TQDM):
        records.extend(parse_hawaii_txt(f, audio_index))
    logger.info(f"  Parsed {len(records)} records")
    return records


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Build manifest by parsing all sources and saving manifest.csv."""
    logger.info("=" * 80)
    logger.info("Starting manifest generation...")
    logger.info("=" * 80)

    logger.info("\nBuilding audio file indices...")
    xavier_index = build_audio_index(XAVIER_DATA_PATH)
    seth_index   = build_audio_index(SETH_DATA_PATH)
    tagus_index  = build_audio_index(TAGUS_DATA_PATH)
    hawaii_index = build_audio_index(HAWAII_DATA_PATH)

    logger.info("\nParsing annotations...")
    all_records = (
        parse_xavier_data(xavier_index)
        + parse_seth_data(seth_index)
        + parse_tagus_data(tagus_index)
        + parse_hawaii_data(hawaii_index)
    )

    if not all_records:
        logger.error("No records found!")
        return

    df = pd.DataFrame(all_records)
    logger.info(f"\nTotal records: {len(df)}")
    logger.info(f"Records by source: {dict(df['source'].value_counts())}")

    # Deployment + timestamp columns (persisted in the output; see
    # deployment_utils.py for the per-source deployment definition).
    unique_files = df[['audio_path', 'source']].drop_duplicates('audio_path')
    file_deployment, file_timestamp = {}, {}
    for row in unique_files.itertuples(index=False):
        file_deployment[row.audio_path] = get_deployment_id(row.audio_path, row.source)
        file_timestamp[row.audio_path]  = parse_timestamp(row.audio_path, row.source)
    df['deployment_id'] = df['audio_path'].map(file_deployment)
    df['timestamp']     = df['audio_path'].map(file_timestamp)

    logger.info("\nReading real audio durations (used for split assignment and "
                "bounds validation)...")
    real_durations = get_audio_durations(df['audio_path'].unique().tolist())

    # New split (Xavier, Seth): 15% of each source's deployments (by total
    # duration) are held out entirely for test. Within the remaining
    # deployments, the chronologically earliest 85% of each deployment's own
    # time is train and the latest 15% is val (val is "the future" relative
    # to train, within a deployment; test is "a place the model has never
    # been"). Deployments with < MIN_FILES_FOR_VAL_SPLIT files go entirely
    # to train — too few files to carve out a meaningful val slice.
    file_split: Dict[str, str] = {}
    short_deployment_counts: Dict[str, int] = defaultdict(int)

    for source in ('xavier', 'seth'):
        src_files = unique_files.loc[unique_files['source'] == source, ['audio_path']].copy()
        src_files['deployment_id'] = src_files['audio_path'].map(file_deployment)
        src_files['timestamp']     = src_files['audio_path'].map(file_timestamp)
        src_files['duration']      = src_files['audio_path'].map(real_durations)
        src_files = src_files.dropna(subset=['duration'])

        deployment_durations = src_files.groupby('deployment_id')['duration'].sum().to_dict()
        test_deployments = select_test_deployments(deployment_durations, TEST_RATIO, RANDOM_SEED)
        total_duration = sum(deployment_durations.values())
        test_duration_frac = (sum(deployment_durations[d] for d in test_deployments)
                               / total_duration) if total_duration else 0.0

        logger.info(f"\n{source}: {len(deployment_durations)} deployments total, "
                    f"{len(test_deployments)} held out for test "
                    f"({100 * test_duration_frac:.1f}% of duration)")

        for dep_id, group in src_files.groupby('deployment_id'):
            if dep_id in test_deployments:
                for p in group['audio_path']:
                    file_split[p] = 'test'
                continue
            if len(group) < MIN_FILES_FOR_VAL_SPLIT:
                short_deployment_counts[source] += 1
                for p in group['audio_path']:
                    file_split[p] = 'train'
                continue
            file_split.update(assign_deployment_train_val(group, VAL_RATIO))

        if short_deployment_counts[source]:
            logger.info(f"  {short_deployment_counts[source]} deployment(s) had "
                        f"< {MIN_FILES_FOR_VAL_SPLIT} files and went entirely to train")

    # Tagus: test deployments are chosen the same way (fixes the "zero test
    # files" gap); train/val within the surviving deployments trusts the
    # original train/validation folder split instead of a chronological
    # duration cut (see FOLD_TAGUS_TEST_INTO_DEPLOYMENT_SPLIT docstring).
    tagus_files = unique_files.loc[unique_files['source'] == 'tagus', ['audio_path']].copy()
    tagus_files['deployment_id'] = tagus_files['audio_path'].map(file_deployment)
    tagus_files['duration']      = tagus_files['audio_path'].map(real_durations)
    tagus_files = tagus_files.dropna(subset=['duration'])

    tagus_deployment_durations = tagus_files.groupby('deployment_id')['duration'].sum().to_dict()
    if FOLD_TAGUS_TEST_INTO_DEPLOYMENT_SPLIT:
        tagus_test_deployments = select_test_deployments(tagus_deployment_durations, TEST_RATIO, RANDOM_SEED)
    else:
        tagus_test_deployments = set()
    total_tagus_duration = sum(tagus_deployment_durations.values())
    tagus_test_frac = (sum(tagus_deployment_durations[d] for d in tagus_test_deployments)
                        / total_tagus_duration) if total_tagus_duration else 0.0

    logger.info(f"\ntagus: {len(tagus_deployment_durations)} deployments total, "
                f"{len(tagus_test_deployments)} held out for test "
                f"({100 * tagus_test_frac:.1f}% of duration)")

    for row in tagus_files.itertuples(index=False):
        if row.deployment_id in tagus_test_deployments:
            file_split[row.audio_path] = 'test'
        else:
            file_split[row.audio_path] = 'val' if 'validation' in Path(row.audio_path).parts else 'train'

    # Hawaii: its own standalone split, never divided into train/val/test —
    # every hawaii file goes straight to split='hawaii' (see
    # SOURCE_EXPECTED_SPLITS docstring above main()).
    hawaii_files = unique_files.loc[unique_files['source'] == 'hawaii', 'audio_path']
    for p in hawaii_files:
        file_split[p] = 'hawaii'
    logger.info(f"\nhawaii: {len(hawaii_files)} files, all assigned to split='hawaii'")

    df['split'] = df['audio_path'].map(file_split)
    if (n_unassigned := df['split'].isna().sum()):
        logger.warning(f"{n_unassigned} row(s) had no split assigned "
                        f"(likely a duration read failure); defaulting to train")
        df['split'] = df['split'].fillna('train')

    # Derived columns
    df['is_fish']    = df['is_fish'].astype('boolean')
    df['duration_s'] = (df['end_s'] - df['begin_s']).round(4)
    df['use_for_detection'] = (
        (df['duration_s'] > 0) &
        (df['duration_s'] <= 2.5) &
        (df['is_fish'].notna())
    )

    # Select and order columns
    df = df[['clip_id', 'source', 'audio_path', 'begin_s', 'end_s', 'duration_s',
             'low_hz', 'high_hz', 'label', 'is_fish', 'use_for_detection', 'split',
             'deployment_id', 'timestamp']]

    # Remove duplicate annotations (same audio file + identical time/freq bounds)
    before = len(df)
    df = df.drop_duplicates(subset=['audio_path', 'begin_s', 'end_s', 'low_hz', 'high_hz'])
    if (removed := before - len(df)):
        logger.info(f"Removed {removed} duplicate annotations")

    # Drop zero/negative duration clips (malformed annotation rows)
    before = len(df)
    df = df[df['duration_s'] > 0]
    if (dropped := before - len(df)):
        logger.info(f"Dropped {dropped} zero/negative-duration clips")

    # Drop annotations whose begin_s/end_s fall outside the resolved audio
    # file's real duration (e.g. an annotation table generated against a
    # longer source recording than the audio file that was kept on disk).
    # Reuses the real_durations dict already fetched above for split assignment.
    logger.info("\nValidating annotation bounds against real audio durations...")
    df['_real_duration_s'] = df['audio_path'].map(real_durations)
    out_of_bounds = (
        df['_real_duration_s'].notna() &
        ((df['begin_s'] < 0) | (df['end_s'] > df['_real_duration_s'] + 1e-3))
    )
    before = len(df)
    if (dropped := out_of_bounds.sum()):
        logger.info("  Dropped out-of-bounds annotations by source:")
        for source, count in df.loc[out_of_bounds, 'source'].value_counts().items():
            logger.info(f"    {source:15s}: {count:6d}")
    df = df[~out_of_bounds].drop(columns=['_real_duration_s'])
    if dropped:
        logger.info(f"Dropped {dropped} out-of-bounds annotations "
                    f"(begin_s/end_s outside resolved file's real duration)")

    df = df.sort_values(['source', 'clip_id']).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\nManifest saved to: {OUTPUT_PATH}")

    # Validate that time/frequency columns are numeric after CSV round-trip
    numeric_cols = ['begin_s', 'end_s', 'low_hz', 'high_hz']
    check = pd.read_csv(OUTPUT_PATH)
    all_ok = True
    for col in numeric_cols:
        if check[col].dtype != np.float64:
            logger.warning(f"VALIDATION: '{col}' dtype is {check[col].dtype}, expected float64")
            all_ok = False
        if (n := check[col].isna().sum()):
            logger.warning(f"VALIDATION: '{col}' contains {n} NaN value(s)")
            all_ok = False
        if (n := check[col].apply(lambda v: isinstance(v, str)).sum()):
            logger.warning(f"VALIDATION: '{col}' contains {n} string value(s)")
            all_ok = False
    logger.info("Validation complete — all numeric columns OK" if all_ok
                else "Validation complete — see warnings above")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("MANIFEST SUMMARY STATISTICS")
    logger.info("=" * 80)

    logger.info(f"\nTotal clips: {len(df)}")

    logger.info("\nClips per source:")
    for source, count in df['source'].value_counts().items():
        logger.info(f"  {source:15s}: {count:8d} ({100 * count / len(df):5.1f}%)")

    logger.info("\nClips per label:")
    for label, count in df['label'].value_counts().items():
        logger.info(f"  {str(label):15s}: {count:8d} ({100 * count / len(df):5.1f}%)")

    logger.info("\nClips per split:")
    for split, count in df['split'].value_counts().items():
        logger.info(f"  {split:15s}: {count:8d} ({100 * count / len(df):5.1f}%)")

    logger.info("\nSource × Split distribution:")
    for line in str(pd.crosstab(df['source'], df['split'], margins=True)).split('\n'):
        logger.info(f"  {line}")

    # ------------------------------------------------------------------
    # SPLIT GUARDRAIL: the bug this replaces produced a val set with zero
    # Xavier files despite Xavier being over half the dataset. Assert (not
    # just log) that every source has data in every split, so a similarly
    # skewed split fails the run instead of silently shipping.
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("SPLIT GUARDRAIL: deployments / files / hours by source x split")
    logger.info("=" * 80)
    logger.info(f"  {'source':10s} {'split':6s} {'deployments':>12s} {'files':>8s} {'hours':>10s}")
    for source in sorted(df['source'].unique()):
        for split in SOURCE_EXPECTED_SPLITS.get(source, ('train', 'val', 'test')):
            sub = df[(df['source'] == source) & (df['split'] == split)]
            n_deployments = sub['deployment_id'].nunique()
            n_files = sub['audio_path'].nunique()
            hours = sub.drop_duplicates('audio_path')['audio_path'].map(real_durations).sum() / 3600
            logger.info(f"  {source:10s} {split:6s} {n_deployments:12d} {n_files:8d} {hours:10.1f}")
            assert n_deployments > 0, f"GUARDRAIL FAILED: {source}/{split} has 0 deployments"
            assert n_files > 0, f"GUARDRAIL FAILED: {source}/{split} has 0 files"

    logger.info("\nFrequency range statistics (Hz):")
    logger.info(f"  Low freq  - Min: {df['low_hz'].min():10.1f}, "
                f"Max: {df['low_hz'].max():10.1f}, Mean: {df['low_hz'].mean():10.1f}")
    logger.info(f"  High freq - Min: {df['high_hz'].min():10.1f}, "
                f"Max: {df['high_hz'].max():10.1f}, Mean: {df['high_hz'].mean():10.1f}")

    logger.info("\nDuration statistics (seconds):")
    logger.info(f"  Min: {df['duration_s'].min():10.3f}, "
                f"Max: {df['duration_s'].max():10.3f}, Mean: {df['duration_s'].mean():10.3f}")
    logger.info("\nClips exceeding duration thresholds:")
    for t in (1, 3, 5, 10, 30, 60):
        n = (df['duration_s'] > t).sum()
        logger.info(f"  > {t:3d}s : {n:6d} ({100 * n / len(df):.2f}%)")
    logger.info("\nClips > 3s by source:")
    long_clips = df[df['duration_s'] > 3]
    for source, count in long_clips['source'].value_counts().items():
        pct = 100 * count / df[df['source'] == source].shape[0]
        logger.info(f"  {source:15s}: {count:6d} ({pct:.2f}% of source)")

    n_det = df['use_for_detection'].sum()
    logger.info(f"\nuse_for_detection=True: {n_det} ({100 * df['use_for_detection'].mean():.1f}%)")
    logger.info("  By source:")
    for source, grp in df.groupby('source'):
        n = grp['use_for_detection'].sum()
        logger.info(f"    {source:15s}: {n:6d} ({100 * n / len(grp):.1f}%)")

    logger.info(f"\nUnique audio files: {df['audio_path'].nunique()}")

    logger.info("\n" + "=" * 80)
    logger.info("Manifest generation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()