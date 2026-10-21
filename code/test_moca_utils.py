"""Synthetic MoCA source and baseline-boundary tests; no production files run."""

import unittest

import pandas as pd

from baseline_utils import VERIFIED_CURATED_SOURCE, prepare_baseline_records
from moca_utils import (
    COMPONENT_MAXIMA,
    KEYS,
    moca_source_audit,
    prepare_raw_moca,
    resolve_moca_sources,
)


def raw_row(patno=1, event="BL", component_total=20, recorded_total=20, **changes):
    row = {
        "PATNO": patno,
        "EVENT_ID": event,
        "REC_ID": f"record-{patno}-{event}",
        "INFODT": "01/2020",
        "MCATOT": recorded_total,
        "MCAVFNUM": 100,
    }
    remaining = component_total
    for column, maximum in COMPONENT_MAXIMA.items():
        row[column] = min(maximum, remaining)
        remaining -= row[column]
    row.update(changes)
    return row


def curated_row(patno=1, event="BL", score=float("nan"), education=float("nan")):
    return {
        "PATNO": patno,
        "EVENT_ID": event,
        "moca": score,
        "EDUCYRS": education,
        "BMI": 24.0,
    }


class MocaSourceTests(unittest.TestCase):
    """Combination of curated and raw MoCA sources."""

    def test_recorded_adjustment_is_not_applied_again(self):
        raw = pd.DataFrame([raw_row(component_total=20, recorded_total=21)])
        cur = pd.DataFrame([curated_row(education=12)])
        row = resolve_moca_sources(cur, raw).iloc[0]
        self.assertEqual(row.moca, 21)
        self.assertEqual(row.moca_raw_component_sum, 20)
        self.assertEqual(row.moca_education_status, "consistent")
        self.assertEqual(row.moca_source, "raw_recorded")
        self.assertEqual(len(COMPONENT_MAXIMA), 26)
        self.assertNotIn("MCAVFNUM", COMPONENT_MAXIMA)

    def test_complete_components_do_not_reconstruct_missing_total(self):
        raw = pd.DataFrame([raw_row(recorded_total=None)])
        row = resolve_moca_sources(pd.DataFrame([curated_row(education=16)]), raw).iloc[
            0
        ]
        self.assertEqual(row.moca_raw_component_sum, 20)
        self.assertEqual(row.moca_raw_reason, "total_missing")
        self.assertTrue(pd.isna(row.moca))

    def test_missing_or_invalid_component_blocks_recorded_total(self):
        for column, value, reason in [
            ("MCACUBE", None, "component_missing"),
            ("MCACUBE", "", "component_missing"),
            ("MCACUBE", 2, "component_invalid"),
            ("MCASER7", 4, "component_invalid"),
            ("MCASNTNC", 3, "component_invalid"),
            ("MCAABSTR", 0.5, "component_invalid"),
            ("MCACUBE", "UR", "component_invalid"),
        ]:
            with self.subTest(column=column, value=value):
                row = prepare_raw_moca(pd.DataFrame([raw_row(**{column: value})])).iloc[
                    0
                ]
                self.assertEqual(row.moca_raw_reason, reason)
                self.assertTrue(pd.isna(row.moca_raw_usable))

    def test_total_range_integrality_and_component_consistency(self):
        for total, reason in [
            (19, "total_component_mismatch"),
            (22, "total_component_mismatch"),
            (20.5, "total_invalid"),
            (-1, "total_invalid"),
            (101, "total_invalid"),
            ("invalid", "total_invalid"),
        ]:
            with self.subTest(total=total):
                row = prepare_raw_moca(
                    pd.DataFrame([raw_row(recorded_total=total)])
                ).iloc[0]
                self.assertEqual(row.moca_raw_reason, reason)
                self.assertTrue(pd.isna(row.moca_raw_usable))
                self.assertEqual(row.moca_raw_MCATOT, total)

    def test_score_ceiling_does_not_receive_an_extra_point(self):
        cur = pd.DataFrame([curated_row(education=12)])
        valid = resolve_moca_sources(
            cur, pd.DataFrame([raw_row(component_total=30, recorded_total=30)])
        ).iloc[0]
        self.assertEqual(valid.moca, 30)
        invalid = resolve_moca_sources(
            cur, pd.DataFrame([raw_row(component_total=30, recorded_total=31)])
        ).iloc[0]
        self.assertTrue(pd.isna(invalid.moca))
        self.assertEqual(invalid.moca_raw_reason, "total_invalid")

    def test_curated_priority_union_and_input_preservation(self):
        cur = pd.DataFrame(
            [curated_row(1, score=25), curated_row(2), curated_row(4, score=27)]
        )
        raw = pd.DataFrame([raw_row(1), raw_row(2), raw_row(3)])
        before_cur, before_raw = cur.copy(deep=True), raw.copy(deep=True)
        result = resolve_moca_sources(cur, raw).set_index("PATNO")
        self.assertEqual(result.index.tolist(), [1, 2, 3, 4])
        self.assertEqual(result.moca.tolist(), [25, 20, 20, 27])
        self.assertEqual(
            result.moca_source.tolist(),
            ["curated", "raw_recorded", "raw_recorded", "curated"],
        )
        self.assertTrue(result.loc[1, "moca_score_disagreement"])
        self.assertTrue(result.loc[1, "moca_threshold_disagreement"])
        self.assertEqual(result.loc[4, "moca_raw_reason"], "raw_row_absent")
        pd.testing.assert_frame_equal(cur, before_cur)
        pd.testing.assert_frame_equal(raw, before_raw)

    def test_education_conflict_blocks_filling_but_not_disagreement_audit(self):
        raw = pd.DataFrame([raw_row(component_total=20, recorded_total=21)])
        cur = pd.DataFrame([curated_row(score=20, education=16)])
        row = resolve_moca_sources(cur, raw).iloc[0]
        self.assertEqual(row.moca, 20)
        self.assertEqual(row.moca_raw_recorded, 21)
        self.assertTrue(pd.isna(row.moca_raw_usable))
        self.assertTrue(row.moca_education_conflict)
        self.assertEqual(row.moca_raw_reason, "education_conflict")
        self.assertTrue(row.moca_score_disagreement)
        self.assertTrue(row.moca_threshold_disagreement)
        cur["moca"] = float("nan")
        self.assertTrue(pd.isna(resolve_moca_sources(cur, raw).iloc[0].moca))

    def test_unknown_education_keeps_recorded_total_with_unknown_label(self):
        result = resolve_moca_sources(
            pd.DataFrame([curated_row()]), pd.DataFrame([raw_row(recorded_total=21)])
        )
        self.assertEqual(result.iloc[0].moca, 21)
        self.assertEqual(result.iloc[0].moca_education_status, "unknown")

    def test_education_temporal_boundaries(self):
        cur = pd.DataFrame(
            [curated_row(1, "BL", education=12), curated_row(2, "SC", education=12)]
        )
        raw = pd.DataFrame(
            [
                raw_row(1, "SC", recorded_total=21),
                raw_row(1, "V04", recorded_total=21),
                raw_row(1, "ST", recorded_total=21),
                raw_row(2, "BL", recorded_total=21),
                raw_row(2, "R06", recorded_total=21),
            ]
        )
        result = resolve_moca_sources(cur, raw).set_index(KEYS)
        self.assertEqual(result.loc[(1, "SC"), "moca_education_source"], "unavailable")
        self.assertEqual(result.loc[(1, "ST"), "moca_education_source"], "unavailable")
        self.assertEqual(result.loc[(1, "V04"), "moca_education_source"], "baseline_BL")
        self.assertEqual(result.loc[(2, "BL"), "moca_education_source"], "earlier_SC")
        self.assertEqual(result.loc[(2, "R06"), "moca_education_source"], "baseline_SC")

    def test_same_visit_education_precedes_baseline_and_future_is_not_used(self):
        cur = pd.DataFrame(
            [
                curated_row(1, "BL", education=12),
                curated_row(1, "V04", education=16),
                curated_row(2, "V04", education=12),
            ]
        )
        raw = pd.DataFrame([raw_row(1, "V04"), raw_row(2, "BL", recorded_total=21)])
        result = resolve_moca_sources(cur, raw).set_index(KEYS)
        self.assertEqual(result.loc[(1, "V04"), "moca"], 20)
        self.assertEqual(result.loc[(1, "V04"), "moca_education_source"], "same_visit")
        self.assertEqual(result.loc[(2, "BL"), "moca_education_source"], "unavailable")

    def test_missing_and_duplicate_keys_are_rejected(self):
        cur, raw = pd.DataFrame([curated_row()]), pd.DataFrame([raw_row()])
        for bad_raw in [
            pd.concat([raw, raw]),
            raw.assign(PATNO=None),
            raw.assign(EVENT_ID=" "),
        ]:
            with self.subTest(source="raw"), self.assertRaisesRegex(ValueError, "keys"):
                resolve_moca_sources(cur, bad_raw)
        for bad_cur in [
            pd.concat([cur, cur]),
            cur.assign(PATNO=None),
            cur.assign(EVENT_ID=""),
        ]:
            with self.subTest(source="curated"), self.assertRaisesRegex(
                ValueError, "keys"
            ):
                resolve_moca_sources(bad_cur, raw)

    def test_required_columns_and_invalid_curated_scores_are_rejected(self):
        cur, raw = pd.DataFrame([curated_row()]), pd.DataFrame([raw_row()])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            resolve_moca_sources(cur, raw.drop(columns="MCACUBE"))
        for score in [31, -1, 20.5, "invalid"]:
            with self.subTest(score=score), self.assertRaisesRegex(
                ValueError, "invalid nonmissing"
            ):
                resolve_moca_sources(cur.assign(moca=score), raw)
        result = resolve_moca_sources(cur.assign(moca=101), raw)
        self.assertEqual(result.iloc[0].moca, 20)

    def test_raw_provenance_is_retained_in_audit(self):
        result = resolve_moca_sources(
            pd.DataFrame([curated_row()]), pd.DataFrame([raw_row()])
        )
        audit = moca_source_audit(result)
        self.assertEqual(audit.iloc[0].moca_raw_REC_ID, "record-1-BL")
        self.assertEqual(audit.iloc[0].moca_raw_INFODT, "01/2020")
        self.assertEqual(audit.iloc[0].moca_raw_MCATOT, 20)
        self.assertEqual(audit.iloc[0].moca_raw_MCASER7, 3)


class MocaBaselineTests(unittest.TestCase):
    """Selection of the baseline MoCA value."""

    def prepare(self, curated, raw):
        return prepare_baseline_records(
            curated, source_path=VERIFIED_CURATED_SOURCE, raw_moca=raw
        )

    def test_only_selected_source_key_can_fill_baseline(self):
        cur = pd.DataFrame(
            [
                curated_row(1),
                curated_row(1, "SC"),
                curated_row(2, "SC"),
                curated_row(3),
                curated_row(4, score=26),
            ]
        )
        raw = pd.DataFrame(
            [
                raw_row(1, "SC"),
                raw_row(2, "SC"),
                raw_row(3, "V04"),
                raw_row(4),
                raw_row(5),
            ]
        )
        result = self.prepare(cur, raw).set_index("PATNO")
        self.assertEqual(result.index.tolist(), [1, 2, 3, 4])
        self.assertEqual(result.EVENT_ID.tolist(), ["BL", "SC", "BL", "BL"])
        self.assertTrue(pd.isna(result.loc[1, "moca"]))
        self.assertTrue(pd.isna(result.loc[3, "moca"]))
        self.assertEqual(result.loc[2, "moca"], 20)
        self.assertEqual(result.loc[4, "moca"], 26)

    def test_full_curated_education_evidence_survives_baseline_selection(self):
        cur = pd.DataFrame([curated_row(1), curated_row(1, "SC", education=12)])
        row = self.prepare(cur, pd.DataFrame([raw_row()])).iloc[0]
        self.assertEqual(row.EVENT_ID, "BL")
        self.assertEqual(row.moca_education_source, "earlier_SC")
        self.assertTrue(row.moca_education_conflict)
        self.assertTrue(pd.isna(row.moca))

    def test_baseline_scores_match_shared_longitudinal_resolution(self):
        cur = pd.DataFrame(
            [
                curated_row(1, education=16),
                curated_row(1, "V04", score=25),
                curated_row(2, "SC", education=12),
            ]
        )
        raw = pd.DataFrame([raw_row(1), raw_row(2, "SC", recorded_total=21)])
        baseline = self.prepare(cur, raw).set_index(KEYS)
        longitudinal = (
            resolve_moca_sources(cur, raw).set_index(KEYS).reindex(baseline.index)
        )
        pd.testing.assert_series_equal(baseline.moca, longitudinal.moca)
        pd.testing.assert_series_equal(baseline.moca_source, longitudinal.moca_source)

    def test_resolving_selected_output_cannot_erase_screening_education_guard(self):
        cur = pd.DataFrame([curated_row(1), curated_row(1, "SC", education=12)])
        raw = pd.DataFrame([raw_row()])
        once = self.prepare(cur, raw)
        self.assertTrue(pd.isna(once.iloc[0].moca))
        with self.assertRaisesRegex(ValueError, "original full curated rows"):
            self.prepare(once, raw)
        bmi_only_repeat = prepare_baseline_records(
            once, source_path=VERIFIED_CURATED_SOURCE
        )
        pd.testing.assert_frame_equal(once, bmi_only_repeat)


if __name__ == "__main__":
    unittest.main()
