import unittest

from metrics.smg_validation import (
    extract_guitar_diagnostic_scores,
    extract_structural_score,
    validate_smg_output,
)


class SmgValidationTests(unittest.TestCase):
    def test_judge_scores_are_extracted_separately(self):
        self.assertEqual(extract_structural_score("Structural Compliance Score: 4.5/5"), 4.5)
        self.assertIsNone(extract_structural_score("Aesthetic Score: 5/5"))
        self.assertIsNone(extract_structural_score("Structural Compliance Score: 6/5"))
        diagnostics = extract_guitar_diagnostic_scores(
            "Layout Score: 4/5\nFingering-Constraint Score: 3/5\n"
            "Structural Compliance Score: 3.5/5"
        )
        self.assertEqual(diagnostics, {"layout_score": 4.0, "fingering_constraint_score": 3.0})

    def test_abc_headers_measure_count_and_meter(self):
        abc = """X:1
T:Structural Test
M:4/4
L:1/4
K:C
C D E F | G A B c |"""
        result = validate_smg_output("staff", abc, expected_measures=2)
        self.assertTrue(result["deterministic_syntax_valid"])
        self.assertTrue(result["deterministic_measure_count_ok"])
        self.assertTrue(result["deterministic_meter_valid"])

    def test_jianpu_fractional_measure_totals(self):
        result = validate_smg_output(
            "jianpu",
            "3(1/4) _5(1/4) 0(1/8) 3(1/16) 4(5/16) | "
            "_6(3/16) __2(1/16) 1(1/4) _6(1/2) |",
            expected_measures=2,
        )
        self.assertTrue(result["deterministic_syntax_valid"])
        self.assertTrue(result["deterministic_measure_count_ok"])
        self.assertTrue(result["deterministic_meter_valid"])
        self.assertEqual(result["deterministic_renderability_status"], "not_assessed")

    def test_jianpu_rejects_incorrect_measure_total(self):
        result = validate_smg_output("jianpu", "1(1/4) 2(1/4) |", expected_measures=1)
        self.assertTrue(result["deterministic_syntax_valid"])
        self.assertFalse(result["deterministic_meter_valid"])

    def test_malformed_duration_denominators_fail_safely(self):
        jianpu = validate_smg_output("jianpu", "1(1/0) |", expected_measures=1)
        self.assertFalse(jianpu["deterministic_syntax_valid"])
        self.assertFalse(jianpu["deterministic_meter_valid"])
        abc = validate_smg_output(
            "staff", "X:1\nT:test\nM:4/0\nL:1/4\nK:C\nC D E F |", expected_measures=1
        )
        self.assertFalse(abc["deterministic_syntax_valid"])

    def test_guitar_tab_checks_width_count_and_fret_span(self):
        tab = """Beat|1 . . . 2 . . . 3 . . . 4 . . . |
 e   |----------------|
 B   |--------1-------|
 G   |----0-----------|
 D   |--2-------------|
 A   |3---------------|
 E   |----------------|"""
        result = validate_smg_output("guitar", tab, expected_measures=1)
        self.assertTrue(result["deterministic_syntax_valid"])
        self.assertTrue(result["deterministic_layout_valid"])
        self.assertTrue(result["deterministic_measure_count_ok"])
        self.assertTrue(result["deterministic_fret_span_valid"])

    def test_guitar_tab_flags_fret_span_above_disclosed_limit(self):
        tab = """e|----------------|
B|0---------------|
G|9---------------|
D|----------------|
A|----------------|
E|----------------|"""
        result = validate_smg_output("guitar", tab, expected_measures=1)
        self.assertFalse(result["deterministic_fret_span_valid"])
        self.assertEqual(result["deterministic_max_fret_span"], 9)


if __name__ == "__main__":
    unittest.main()
