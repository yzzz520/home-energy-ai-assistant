from __future__ import annotations

import unittest

from numeric_check import check_answer_numbers


class NumericCheckTest(unittest.TestCase):
    def test_rounded_tool_number_passes(self) -> None:
        audit = check_answer_numbers(
            "总成本为 209.24 AUD，SOC 为 0.500001。",
            [
                {
                    "total_cost": 209.23904,
                    "final_soc": 0.500001,
                }
            ],
        )
        self.assertTrue(audit.valid)

    def test_invented_number_fails(self) -> None:
        audit = check_answer_numbers(
            "预计节省 99.99 AUD。",
            [{"total_cost": 209.23904}],
        )
        self.assertFalse(audit.valid)
        self.assertEqual(audit.unmatched_numbers, ["99.99"])

    def test_numbers_inside_tool_strings_are_allowed(self) -> None:
        audit = check_answer_numbers(
            "规则要求在预计收益不足 20% 时回退。",
            [{"content": "规则要求在预计收益不足20%时回退。"}],
        )
        self.assertTrue(audit.valid)


if __name__ == "__main__":
    unittest.main()
