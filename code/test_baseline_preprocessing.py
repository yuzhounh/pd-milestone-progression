"""Unit tests for baseline_preprocessing.py."""

import unittest

import numpy as np
import pandas as pd

from baseline_preprocessing import RACE_DUMMY_COLUMNS, fit_baseline_preprocessor


class BaselinePreprocessingTests(unittest.TestCase):
    """Training-only imputation, scaling and race encoding."""

    def test_race_is_dummy_encoded_with_white_reference(self):
        train = pd.DataFrame(
            {"age": [50, 60, 70, 80], "race": [1, 2, 3, 4], "binary": [0, 1, 0, 1]}
        )
        validation = pd.DataFrame(
            {"age": [55, 65], "race": [1, 4], "binary": [1, np.nan]}
        )
        result = fit_baseline_preprocessor(train, validation)
        self.assertNotIn("race", result.train_filled)
        self.assertEqual(
            [column for column in result.train_filled if column.startswith("race_")],
            RACE_DUMMY_COLUMNS,
        )
        self.assertEqual(result.train_filled.loc[0, RACE_DUMMY_COLUMNS].sum(), 0)
        self.assertEqual(result.train_filled.loc[3, "race_Other"], 1)
        self.assertEqual(result.validation_filled.loc[0, RACE_DUMMY_COLUMNS].sum(), 0)

    def test_race_missingness_is_imputed_from_training_only(self):
        train = pd.DataFrame({"race": [2, 2, 1, np.nan], "x": [1, 2, 3, 4]})
        validation = pd.DataFrame({"race": [np.nan, 4], "x": [5, 6]})
        result = fit_baseline_preprocessor(train, validation)
        self.assertEqual(result.race_imputation_code, 2)
        self.assertEqual(result.validation_filled.loc[0, "race_Black"], 1)
        self.assertEqual(result.validation_filled.loc[1, "race_Other"], 1)

    def test_invalid_race_code_is_rejected(self):
        train = pd.DataFrame({"race": [1, 5], "x": [1, 2]})
        validation = pd.DataFrame({"race": [1], "x": [3]})
        with self.assertRaisesRegex(ValueError, "Invalid race codes"):
            fit_baseline_preprocessor(train, validation)

    def test_validation_values_do_not_change_training_imputation(self):
        train = pd.DataFrame({"race": [1, 1, np.nan], "x": [1.0, np.nan, 5.0]})
        validation = pd.DataFrame({"race": [4, 4], "x": [1000.0, np.nan]})
        result = fit_baseline_preprocessor(train, validation)
        self.assertEqual(result.race_imputation_code, 1)
        self.assertEqual(result.train_filled.loc[1, "x"], 1.0)
        self.assertEqual(result.validation_filled.loc[1, "x"], 1.0)


if __name__ == "__main__":
    unittest.main()
