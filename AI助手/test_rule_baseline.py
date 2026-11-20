from __future__ import annotations

import unittest
from pathlib import Path

from rule_baseline import evaluate_seed_cases, extract_slots, predict


class RuleBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent
        cls.predictions, cls.metrics = evaluate_seed_cases(
            root / "data" / "eval_seed.csv"
        )

    def test_slot_extraction(self) -> None:
        slots = extract_slots(
            "为什么客户1在2011-06-15 18:00放电"
        )
        self.assertEqual(slots["customer_id"], 1)
        self.assertEqual(slots["date"], "2011-06-15")
        self.assertEqual(
            slots["interval_start"],
            "2011-06-15T18:00:00",
        )

    def test_what_if_slots(self) -> None:
        slots = extract_slots("如果客户1明天没有光伏会怎么样")
        self.assertEqual(slots["scenario"], {"pv": 0})
        self.assertEqual(slots["customer_id"], 1)

    def test_seed_intent_accuracy(self) -> None:
        self.assertGreaterEqual(self.metrics["intent_accuracy"], 0.95)
        self.assertGreaterEqual(self.metrics["slot_exact_match_rate"], 0.95)
        self.assertGreaterEqual(self.metrics["slot_micro_f1"], 0.95)

    def test_unknown_weather_is_out_of_scope(self) -> None:
        self.assertEqual(
            predict("今天天气怎么样").intent_id,
            "unknown_or_out_of_scope",
        )


if __name__ == "__main__":
    unittest.main()
