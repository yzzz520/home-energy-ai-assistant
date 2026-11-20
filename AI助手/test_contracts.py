from __future__ import annotations

import unittest

from contract_validator import ContractValidator
from scope import UserScopeResolver


class UserScopeResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = UserScopeResolver.from_config()

    def test_known_customer(self) -> None:
        result = self.resolver.resolve(customer_id=1, dataset_id=None)
        self.assertEqual(result.user_scope, "existing_300")
        self.assertEqual(result.reason_code, "known_customer")

    def test_unknown_customer_without_upload(self) -> None:
        result = self.resolver.resolve(customer_id=999, dataset_id=None)
        self.assertEqual(result.user_scope, "unsupported")
        self.assertEqual(result.reason_code, "unknown_customer_no_upload")

    def test_unvalidated_upload(self) -> None:
        result = self.resolver.resolve(
            customer_id=None,
            dataset_id="upload-1",
        )
        self.assertEqual(result.user_scope, "unsupported")
        self.assertTrue(result.requires_upload_validation)

    def test_validated_cold_start(self) -> None:
        result = self.resolver.resolve(
            customer_id=None,
            dataset_id="upload-1",
            dataset_validated=True,
        )
        self.assertEqual(result.user_scope, "cold_start")
        self.assertFalse(result.requires_upload_validation)


class ContractValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validator = ContractValidator()

    def test_existing_user_schedule_contract(self) -> None:
        result = self.validator.validate(
            intent_id="schedule_query",
            user_scope="existing_300",
            slots={"customer_id": 1, "date": "2011-06-15"},
            requested_tools=["get_schedule"],
        )
        self.assertTrue(result.valid)
        self.assertFalse(result.needs_clarification)
        self.assertEqual(
            result.allowed_tools,
            ["get_schedule", "get_energy_summary"],
        )

    def test_missing_date_needs_clarification(self) -> None:
        result = self.validator.validate(
            intent_id="schedule_query",
            user_scope="existing_300",
            slots={"customer_id": 1},
        )
        self.assertFalse(result.valid)
        self.assertTrue(result.needs_clarification)
        self.assertIn("date", result.missing_slots)

    def test_cold_start_uses_dataset_identity(self) -> None:
        result = self.validator.validate(
            intent_id="load_forecast_query",
            user_scope="cold_start",
            slots={
                "dataset_id": "upload-1",
                "date": "2011-06-15",
                "target": "load",
            },
            requested_tools=["get_forecast"],
        )
        self.assertTrue(result.valid)

    def test_cold_start_cannot_use_existing_customer_identity(self) -> None:
        result = self.validator.validate(
            intent_id="load_forecast_query",
            user_scope="cold_start",
            slots={
                "customer_id": 1,
                "date": "2011-06-15",
                "target": "load",
            },
        )
        self.assertFalse(result.valid)
        self.assertIn("dataset_id", result.missing_slots)

    def test_invalid_target_and_datetime_are_rejected(self) -> None:
        result = self.validator.validate(
            intent_id="schedule_explanation",
            user_scope="existing_300",
            slots={
                "customer_id": 1,
                "date": "2011-06-15",
                "interval_start": "2011-06-15T18:00:00+10:00",
                "target": "battery",
            },
        )
        self.assertFalse(result.valid)
        self.assertTrue(any(item.startswith("TIMEZONE_NOT_ALLOWED") for item in result.errors))
        self.assertTrue(any(item.startswith("INVALID_ENUM") for item in result.errors))

    def test_scope_not_allowed(self) -> None:
        result = self.validator.validate(
            intent_id="new_user_guidance",
            user_scope="existing_300",
            slots={},
        )
        self.assertFalse(result.valid)
        self.assertTrue(any(item.startswith("USER_SCOPE_NOT_ALLOWED") for item in result.errors))

    def test_tool_plan_permission(self) -> None:
        result = self.validator.validate(
            intent_id="status_query",
            user_scope="existing_300",
            slots={"customer_id": 1},
            requested_tools=["simulate_scenario"],
        )
        self.assertFalse(result.valid)
        self.assertTrue(any(item.startswith("TOOL_NOT_ALLOWED") for item in result.errors))

    def test_unknown_slot_is_rejected(self) -> None:
        result = self.validator.validate(
            intent_id="status_query",
            user_scope="existing_300",
            slots={"customer_id": 1, "unexpected": "x"},
        )
        self.assertFalse(result.valid)
        self.assertIn("UNKNOWN_SLOT:unexpected", result.errors)


if __name__ == "__main__":
    unittest.main()
