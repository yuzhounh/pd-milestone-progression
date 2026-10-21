"""Unit tests for milestone_history_utils.py."""

import unittest

import numpy as np
import pandas as pd

from lca_utils import build_discrete_time_matrix, subtype_posterior_column
from milestone_history_utils import (
    build_domain_history,
    collapse_identical_visit_rows,
    mask_st_negatives,
    partiv_structural_negatives,
    resolve_visit_times,
)


class MilestoneHistoryTests(unittest.TestCase):
    """Event-history construction from positive, negative and unknown visit states."""

    def test_identical_duplicate_visit_rows_are_collapsed(self):
        frame = pd.DataFrame(
            {
                "PATNO": [548922, 548922],
                "EVENT_ID": ["BL", "BL"],
                "SCAU8": [0, 0],
                "SCAU9": [1, 1],
                "REC_ID": ["IA1056800", "IA1056801"],
            }
        )
        result = collapse_identical_visit_rows(frame, ["SCAU8", "SCAU9"], "SCOPA-AUT")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].REC_ID, "IA1056800")

    def test_conflicting_duplicate_visit_rows_are_rejected(self):
        frame = pd.DataFrame(
            {
                "PATNO": [548922, 548922],
                "EVENT_ID": ["BL", "BL"],
                "SCAU8": [0, 1],
                "SCAU9": [1, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "disagree on analysis values"):
            collapse_identical_visit_rows(frame, ["SCAU8", "SCAU9"], "SCOPA-AUT")

    def history(self, visits, values):
        frame = pd.DataFrame(
            {
                "PATNO": [1] * len(visits),
                "EVENT_ID": visits,
                "time": [
                    {
                        "BL": 0,
                        "V04": 1,
                        "V06": 2,
                        "V07": 2.5,
                        "R06": 2.5,
                        "V08": 3,
                        "V10": 4,
                    }[visit]
                    for visit in visits
                ],
            }
        )
        status = pd.Series(values, dtype="boolean")
        return build_domain_history(frame, status, "Walking").iloc[0]

    def test_remote_positive_is_retained_at_subannual_time(self):
        row = self.history(["BL", "V04", "V06", "R06"], [False, False, False, True])
        self.assertTrue(row.event)
        self.assertEqual(row.event_time, 2.5)
        self.assertEqual(row.event_interval, 3)
        self.assertEqual(row.first_positive_visits, "R06")

    def test_same_time_standard_and_remote_positive_wins(self):
        row = self.history(
            ["BL", "V04", "V06", "V07", "R06"],
            [False, False, False, False, True],
        )
        self.assertTrue(row.event)
        self.assertEqual(row.event_time, 2.5)

    def test_missing_annual_visit_truncates_later_event(self):
        row = self.history(["BL", "V06", "V08"], [False, False, True])
        self.assertFalse(row.event)
        self.assertTrue(row.event_lost_after_gap)
        self.assertEqual(row.first_gap_year, 1)
        self.assertEqual(row.censor_time, 0)
        self.assertEqual(row.observable_cells, 0)

    def test_missing_annual_after_event_is_irrelevant(self):
        row = self.history(["BL", "V04", "V06"], [False, True, pd.NA])
        self.assertTrue(row.event)
        self.assertEqual(row.event_time, 1)
        self.assertTrue(np.isnan(row.first_gap_year))

    def test_unknown_annual_state_truncates_domain_only(self):
        row = self.history(["BL", "V04", "V06"], [False, pd.NA, False])
        self.assertFalse(row.event)
        self.assertEqual(row.first_gap_reason, "unknown_state")
        self.assertEqual(row.censor_time, 0)

    def test_st_negative_is_masked_but_positive_is_retained(self):
        frame = pd.DataFrame({"EVENT_ID": ["BL", "ST"]})
        states = pd.DataFrame(
            {
                "Walking": pd.Series([False, False], dtype="boolean"),
                "ADL": pd.Series([False, True], dtype="boolean"),
            }
        )
        result = mask_st_negatives(frame, states)
        self.assertTrue(pd.isna(result.loc[1, "Walking"]))
        self.assertTrue(bool(result.loc[1, "ADL"]))

    def test_st_time_uses_part3_months_and_drops_beyond_six_years(self):
        source = pd.DataFrame(
            {
                "PATNO": [1, 1, 2, 2, 3, 3],
                "EVENT_ID": ["BL", "ST", "BL", "ST", "BL", "ST"],
            }
        )
        part3 = pd.DataFrame(
            {
                "PATNO": [1, 1, 2, 2, 3, 3],
                "EVENT_ID": ["BL", "ST", "BL", "ST", "BL", "ST"],
                "INFODT": [
                    "01/2020",
                    "07/2020",
                    "01/2020",
                    "06/2026",
                    "01/2020",
                    "01/2020",
                ],
            }
        )
        result = resolve_visit_times(source, part3)
        self.assertEqual(result.iloc[1], 0.5)
        self.assertTrue(np.isnan(result.iloc[3]))
        self.assertTrue(np.isnan(result.iloc[5]))

    def test_partiv_structural_negative_requires_no_prior_evidence(self):
        source = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "NP4WDYSK": [np.nan, np.nan],
                "NP4DYSKI": [np.nan, np.nan],
                "NP4FLCTI": [np.nan, np.nan],
                "NP4FLCTX": [np.nan, np.nan],
            }
        )
        part3 = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "PDTRTMNT": [0, 0],
                "PDMEDYN": [0, 1],
                "DBSYN": [0, 0],
                "PDSTATE": [np.nan, np.nan],
                "INFODT": ["01/2020", "01/2021"],
            }
        )
        part4 = source.copy()
        curated = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "PDTRTMNT": [0, 0],
                "LEDD": [0, 0],
            }
        )
        result = partiv_structural_negatives(
            source, part3, part4, curated, pd.Series([0.0, 1.0])
        )
        self.assertTrue(bool(result.loc[0, "structural_negative"]))
        self.assertFalse(bool(result.loc[1, "structural_negative"]))

    def test_partiv_prior_treatment_blocks_later_structural_negative(self):
        source = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "NP4WDYSK": [np.nan, np.nan],
                "NP4DYSKI": [np.nan, np.nan],
                "NP4FLCTI": [np.nan, np.nan],
                "NP4FLCTX": [np.nan, np.nan],
            }
        )
        part3 = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "PDTRTMNT": [1, 0],
                "PDMEDYN": [1, 0],
                "DBSYN": [0, 0],
                "PDSTATE": ["ON", np.nan],
                "INFODT": ["01/2020", "01/2021"],
            }
        )
        curated = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "PDTRTMNT": [1, 0],
                "LEDD": [100, 0],
            }
        )
        result = partiv_structural_negatives(
            source, part3, source.copy(), curated, pd.Series([0.0, 1.0])
        )
        self.assertFalse(bool(result.loc[1, "structural_negative"]))
        self.assertTrue(bool(result.loc[1, "prior_evidence"]))

    def test_partiv_future_treatment_does_not_reclassify_earlier_visit(self):
        source = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "NP4WDYSK": [np.nan, np.nan],
                "NP4DYSKI": [np.nan, np.nan],
                "NP4FLCTI": [np.nan, np.nan],
                "NP4FLCTX": [np.nan, np.nan],
            }
        )
        part3 = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "PDTRTMNT": [0, 1],
                "PDMEDYN": [0, 1],
                "DBSYN": [0, 0],
                "PDSTATE": [np.nan, "ON"],
                "INFODT": ["01/2020", "01/2021"],
            }
        )
        curated = pd.DataFrame(
            {
                "PATNO": [1, 1],
                "EVENT_ID": ["BL", "V04"],
                "PDTRTMNT": [0, 1],
                "LEDD": [0, 100],
            }
        )
        result = partiv_structural_negatives(
            source, part3, source.copy(), curated, pd.Series([0.0, 1.0])
        )
        self.assertTrue(bool(result.loc[0, "structural_negative"]))
        self.assertFalse(bool(result.loc[0, "prior_evidence"]))

    def test_partiv_unrateable_value_is_not_structural_negative(self):
        source = pd.DataFrame(
            {
                "PATNO": [1],
                "EVENT_ID": ["BL"],
                "NP4WDYSK": [np.nan],
                "NP4DYSKI": [np.nan],
                "NP4FLCTI": [np.nan],
                "NP4FLCTX": [np.nan],
            }
        )
        part3 = pd.DataFrame(
            {
                "PATNO": [1],
                "EVENT_ID": ["BL"],
                "PDTRTMNT": [0],
                "PDMEDYN": [0],
                "DBSYN": [0],
                "PDSTATE": [np.nan],
                "INFODT": ["01/2020"],
            }
        )
        part4 = source.astype(
            {
                column: "object"
                for column in ["NP4WDYSK", "NP4DYSKI", "NP4FLCTI", "NP4FLCTX"]
            }
        )
        part4.loc[0, "NP4WDYSK"] = "UR"
        curated = pd.DataFrame(
            {"PATNO": [1], "EVENT_ID": ["BL"], "PDTRTMNT": [0], "LEDD": [0]}
        )
        result = partiv_structural_negatives(
            source, part3, part4, curated, pd.Series([0.0])
        )
        self.assertFalse(bool(result.loc[0, "structural_negative"]))

    def test_lca_matrix_uses_each_domains_own_censoring_horizon(self):
        frame = pd.DataFrame(
            {
                "Time_Walking": [7, 7],
                "Censor_Walking": [1.0, 3.5],
                "Time_ADL": [7, 2],
                "Censor_ADL": [4.0, 1.5],
            },
            index=pd.Index([1, 2], name="PATNO"),
        )
        matrix = build_discrete_time_matrix(frame, ["Time_Walking", "Time_ADL"])
        self.assertEqual(matrix.loc[1, "Walking_interval_1"], 0)
        self.assertTrue(pd.isna(matrix.loc[1, "Walking_interval_2"]))
        self.assertEqual(matrix.loc[1, "ADL_interval_4"], 0)
        self.assertEqual(matrix.loc[2, "Walking_interval_3"], 0)
        self.assertEqual(matrix.loc[2, "ADL_interval_1"], 0)
        self.assertEqual(matrix.loc[2, "ADL_interval_2"], 1)

    def test_named_posterior_lookup_does_not_assume_class_zero(self):
        frame = pd.DataFrame(
            {"Posterior_Class_0_Slow": [0.8], "Posterior_Class_1_Rapid": [0.2]}
        )
        self.assertEqual(
            subtype_posterior_column(frame, "Rapid"), "Posterior_Class_1_Rapid"
        )


if __name__ == "__main__":
    unittest.main()
