import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from Complete_Function import append_df_to_excel

class TestCompleteFunction(unittest.TestCase):
    def setUp(self):
        self.filename = "test_output.xlsx"
        self.df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_append_df_to_excel_new_file(self):
        # Test creating a new file
        append_df_to_excel(self.filename, self.df, sheet_name='Sheet1')
        self.assertTrue(os.path.exists(self.filename))

        # Verify contents
        read_df = pd.read_excel(self.filename, sheet_name='Sheet1', index_col=0)
        pd.testing.assert_frame_equal(self.df, read_df)

    def test_append_df_to_excel_append(self):
        # Create initial file
        append_df_to_excel(self.filename, self.df, sheet_name='Sheet1')

        # Append second dataframe
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        append_df_to_excel(self.filename, df2, sheet_name='Sheet1', header=False)

        # Verify contents
        read_df = pd.read_excel(self.filename, sheet_name='Sheet1', index_col=0)
        expected_df = pd.concat([self.df, df2], ignore_index=True)
        # Note: the index of read_df won't be clean because of how appending works,
        # but the values should match
        np.testing.assert_array_equal(read_df.values, expected_df.values)

    def test_append_df_to_excel_truncate(self):
        # Create initial file
        append_df_to_excel(self.filename, self.df, sheet_name='Sheet1')

        # Truncate and write new dataframe
        df2 = pd.DataFrame({'C': [9, 10], 'D': [11, 12]})
        append_df_to_excel(self.filename, df2, sheet_name='Sheet1', truncate_sheet=True)

        # Verify contents
        read_df = pd.read_excel(self.filename, sheet_name='Sheet1', index_col=0)
        pd.testing.assert_frame_equal(df2, read_df)

if __name__ == '__main__':
    unittest.main()
