"""Synthetic checks for verified baseline corrections; no files or models run."""

import unittest

import pandas as pd

from baseline_utils import (
    BMI_AUDIT_COLUMNS,
    BMI_CORRECTION_METADATA,
    BMI_ORIGINAL_COLUMN,
    VERIFIED_CURATED_SOURCE,
    baseline_correction_audit,
    normalize_curated_columns,
    prepare_baseline_records,
    select_baseline_records,
)


class VerifiedBaselineCorrectionTests(unittest.TestCase):
    """Corrections applied to verified baseline values."""

    def setUp(self):
        self.curated = pd.DataFrame(
            {
                "PATNO": [137482, 2, 3, 4, 137482],
                "EVENT_ID": ["BL", "BL", "BL", "SC", "SC"],
                "BMI": [56200.0, 73.46, 56200.0, float("nan"), 23.0],
                "APOE_e4": [2.0, 0.0, 1.0, float("nan"), 2.0],
                "LEDD": [0, 1900, 100, 0, 0],
            }
        )

    def prepare(self, frame):
        return prepare_baseline_records(frame, source_path=VERIFIED_CURATED_SOURCE)

    def test_only_verified_bmi_changes_and_input_is_not_mutated(self):
        original = self.curated.copy(deep=True)
        selected = select_baseline_records(self.curated)
        result = self.prepare(self.curated)
        expected = selected.copy()
        expected.loc[expected["PATNO"].eq(137482), "BMI"] = float("nan")
        pd.testing.assert_frame_equal(result[selected.columns], expected)
        pd.testing.assert_frame_equal(self.curated, original)
        self.assertEqual(result.loc[result["PATNO"].eq(2), "BMI"].item(), 73.46)
        self.assertEqual(result.loc[result["PATNO"].eq(3), "BMI"].item(), 56200.0)

    def test_dosage_and_missing_apoe_are_preserved(self):
        selected = select_baseline_records(self.curated)
        result = self.prepare(self.curated)
        pd.testing.assert_series_equal(result["APOE_e4"], selected["APOE_e4"])
        self.assertEqual(set(result["APOE_e4"].dropna()), {0.0, 1.0, 2.0})
        self.assertEqual(result["APOE_e4"].isna().sum(), 1)

    def test_audit_preserves_value_evidence_and_version(self):
        audit = baseline_correction_audit(self.prepare(self.curated))
        self.assertEqual(len(audit), 1)
        row = audit.iloc[0]
        self.assertEqual(row["PATNO"], 137482)
        self.assertEqual(row["baseline_source_event"], "BL")
        self.assertEqual(row[BMI_ORIGINAL_COLUMN], 56200.0)
        self.assertTrue(pd.isna(row["BMI"]))
        for column, value in BMI_CORRECTION_METADATA.items():
            self.assertEqual(row[column], value)

    def test_repeat_application_is_idempotent(self):
        once = self.prepare(self.curated)
        twice = self.prepare(once)
        pd.testing.assert_frame_equal(once, twice)
        pd.testing.assert_frame_equal(
            baseline_correction_audit(once), baseline_correction_audit(twice)
        )

    def test_screening_is_not_used_to_replace_invalid_baseline(self):
        result = self.prepare(self.curated)
        target = result.loc[result["PATNO"].eq(137482)].iloc[0]
        self.assertEqual(target["EVENT_ID"], "BL")
        self.assertTrue(pd.isna(target["BMI"]))

    def test_screening_only_and_absent_target_are_not_corrected(self):
        for frame in (
            self.curated.loc[self.curated["EVENT_ID"].eq("SC")],
            self.curated.loc[self.curated["PATNO"].ne(137482)],
        ):
            with self.subTest(participants=frame["PATNO"].tolist()):
                result = self.prepare(frame)
                selected = select_baseline_records(frame)
                pd.testing.assert_frame_equal(result[selected.columns], selected)
                self.assertTrue(baseline_correction_audit(result).empty)

    def test_changed_source_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires curated source"):
            prepare_baseline_records(self.curated, source_path="new_release.xlsx")

    def test_changed_or_missing_original_value_is_rejected(self):
        for value in (26.0, 56201.0, float("nan"), "invalid"):
            with self.subTest(value=value):
                frame = self.curated.astype({"BMI": object})
                frame.loc[0, "BMI"] = value
                with self.assertRaisesRegex(ValueError, "differs from audited value"):
                    self.prepare(frame)

    def test_incomplete_or_tampered_provenance_is_rejected(self):
        result = self.prepare(self.curated)
        with self.assertRaisesRegex(ValueError, "Incomplete BMI correction"):
            self.prepare(result.drop(columns=BMI_AUDIT_COLUMNS[-1]))
        result.loc[result["PATNO"].eq(137482), "BMI_correction_rule"] = "other_v2"
        with self.assertRaisesRegex(ValueError, "differs from audited value"):
            self.prepare(result)

    def test_audit_on_unaudited_record_is_rejected(self):
        result = self.prepare(self.curated)
        result.loc[result["PATNO"].eq(2), BMI_ORIGINAL_COLUMN] = 73.46
        with self.assertRaisesRegex(ValueError, "unaudited record"):
            self.prepare(result)

    def test_float_storage_tolerance_does_not_lose_original(self):
        frame = self.curated.copy()
        frame.loc[0, "BMI"] = 56199.99999999999
        result = self.prepare(frame)
        self.assertEqual(
            baseline_correction_audit(result)[BMI_ORIGINAL_COLUMN].item(),
            frame.loc[0, "BMI"],
        )

    def test_duplicate_source_records_remain_rejected(self):
        duplicate = pd.concat([self.curated, self.curated.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate participant/source"):
            self.prepare(duplicate)

    def test_current_curated_columns_are_normalized_to_stable_names(self):
        frame = pd.DataFrame(
            {
                "MIA_CAUDATE_BILAT": [1.0],
                "MIA_PUTAMEN_BILAT": [2.0],
                "MIA_STRIATUM_BILAT": [3.0],
                "MIA_LOWPUT_EXPECTED": [4.0],
            }
        )
        normalized = normalize_curated_columns(frame)
        self.assertEqual(
            normalized.columns.tolist(),
            [
                "MIA_CAUDATE_mean",
                "MIA_PUTAMEN_mean",
                "MIA_STRIATUM_mean",
                "mia_lowput_expected",
            ],
        )
        self.assertEqual(normalized.iloc[0].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_ambiguous_curated_aliases_are_rejected(self):
        frame = pd.DataFrame({"MIA_STRIATUM_BILAT": [0.8], "MIA_STRIATUM_mean": [0.8]})
        with self.assertRaisesRegex(ValueError, "ambiguous renamed columns"):
            normalize_curated_columns(frame)


if __name__ == "__main__":
    unittest.main()
