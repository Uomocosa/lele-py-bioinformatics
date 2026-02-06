import lele

THIS_FOLDER = lele.P(__file__).parent
HELPER_DIR = THIS_FOLDER/'__HELPER_DIR__'

assert HELPER_DIR.exists()

def test_():
    pass
