import lele

THIS_FOLDER = lele.P(__file__).parent
HELPER_DIR = THIS_FOLDER/'__HELPER_DIR__'
MIN_GPT_CONFIG_FILE = HELPER_DIR / "config.jsonc"
CHECKPOINT_FOLDER = HELPER_DIR /  "checkpoints"

assert HELPER_DIR.exists()
assert MIN_GPT_CONFIG_FILE.exists()
assert CHECKPOINT_FOLDER.exists()

def test_():
    pass
