from __future__ import annotations

import unittest

from assistant_pipeline import DeterministicAssistant, evaluate_pipeline


class AssistantPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.assistant = DeterministicAssistant()

    def test_schedule_query_end_to_end(self) -> None:
        result = self.assistant.process(
            "客户1在2011-05-16的充放电计划"
        )
        self.assertEqual(result.decision.intent_id, "schedule_query")
        self.assertEqual(result.outcome, "answer")
        self.assertTrue(result.numeric_audit.valid)
        self.assertTrue(result.tool_results)

    def test_missing_slot_clarification(self) -> None:
        result = self.assistant.process("查看明天计划")
        self.assertEqual(result.outcome, "clarify")
        self.assertIn("customer_id", result.contract.missing_slots)
        self.assertTrue(result.numeric_audit.valid)

    def test_device_control_refusal(self) -> None:
        result = self.assistant.process("帮我直接关闭家里的电池")
        self.assertEqual(result.outcome, "refuse")
        self.assertTrue(result.numeric_audit.valid)

    def test_unsupported_tool_is_reported(self) -> None:
        result = self.assistant.process(
            "客户1在2011-05-16的光伏预测是什么"
        )
        self.assertEqual(result.outcome, "tool_error")
        self.assertIn("尚未接入", result.answer)
        self.assertTrue(result.numeric_audit.valid)

    def test_seed_metrics(self) -> None:
        _, metrics = evaluate_pipeline(assistant=self.assistant)
        self.assertGreaterEqual(metrics["intent_accuracy"], 0.95)
        self.assertGreaterEqual(metrics["slot_exact_match_rate"], 0.95)
        self.assertGreaterEqual(metrics["numeric_consistency_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
