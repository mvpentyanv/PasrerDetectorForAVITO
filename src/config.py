# Общие настройки проекта: пути, сиды, параметры валидации.
# Всё в одном месте, чтобы эксперименты и финальный пайплайн гарантированно брали одно и то же.
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
EVENTS_PATH = DATA_DIR / "events.csv.gz"
SAMPLE_SUB_PATH = ROOT / "sample_submission.csv"
SUBMISSION_PATH = ROOT / "submission.csv"

SEED = 42
N_FOLDS = 5
TARGET_RECALL = 0.7

META_TIME_COLS = ["cookie_created_at", "window_start_ts", "window_end_ts"]
# разброс P@R между сидами около 0.005, поэтому финальный LightGBM усредняю по пяти сидам
SEEDS = [42, 1, 2, 3, 4]
