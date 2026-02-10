import lele

THIS_DIR = lele.P(__file__).parent
HELPER_DIR = THIS_DIR/"__HELPER_DIR__"
FPSCORES_JSON = HELPER_DIR/"fpscores.json"
FPSCORES = None

assert THIS_DIR.exists()
assert HELPER_DIR.exists()
assert FPSCORES_JSON.exists()

def test_():
    pass
