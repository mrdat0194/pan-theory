import unittest
from unittest.mock import patch, MagicMock

# The problem is `from MLModel.model.res.plot_lib import _get_clr` in setUp
# causes it to try importing matplotlib multiple times which has native extensions
# failing. We should just import it once at module level, but we have to patch sys.modules
# around the import block safely, then we can run tests.

with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'IPython': MagicMock(),
    'IPython.display': MagicMock()
}):
    from MLModel.model.res.plot_lib import _get_clr

class TestPlotLib(unittest.TestCase):

    def test_get_clr_normal(self):
        self.assertEqual(_get_clr(0), '#85c2e1')
        self.assertEqual(_get_clr(0.5), '#f9e8e8')

    def test_get_clr_out_of_bounds_negative(self):
        self.assertEqual(_get_clr(-100), '#85c2e1')

    def test_get_clr_out_of_bounds_positive(self):
        self.assertEqual(_get_clr(100), '#f42e2e')

if __name__ == '__main__':
    unittest.main()
