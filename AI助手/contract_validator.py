from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContractValidationResult:
    intent_id: str
    user_scope: str
    valid: bool
    needs_clarification: bool
    slots: dict[str, Any]
    missing_slots: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)


class ContractValidator:
    def __init__(
        self,
        *,
        intents_path: str | Path | None = None,
        tools_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent
        self.intents_path = (
            Path(intents_path)
            if intents_path is not None
            else root / "contracts" / "intents.json"
        )
        self.tools_path = (
            Path(tools_path)
            if tools_path is not None
            else root / "contracts" / "tools.json"
        )
        self.intents_contract = json.loads(
            self.intents_path.read_text(encoding="utf-8")
        )
        self.tools_contract = json.loads(
            self.tools_path.read_text(encoding="utf-8")
        )
        self.intents = {
            item["id"]: item for item in self.intents_contract["intents"]
        }
        self.slot_definitions = self.intents_contract["slot_definitions"]
        self.tools = set(self.tools_contract["tools"])

    def validate(
        self,
        *,
        intent_id: str,
        user_scope: str,
        slots: dict[str, Any],
        requested_tools: list[str] | None = None,
    ) -> ContractValidationResult:
        errors: list[str] = []
        missing_slots: list[str] = []
        normalized_slots: dict[str, Any] = {}

        intent = self.intents.get(intent_id)
        if intent is None:
            return ContractValidationResult(
                intent_id=intent_id,
                user_scope=user_scope,
                valid=False,
                needs_clarification=False,
                slots={},
                errors=[f"UNKNOWN_INTENT:{intent_id}"],
            )

        if user_scope not in self.intents_contract["user_scopes"]:
            errors.append(f"UNKNOWN_USER_SCOPE:{user_scope}")
        elif user_scope not in intent["user_scopes"]:
            errors.append(f"USER_SCOPE_NOT_ALLOWED:{intent_id}:{user_scope}")

        extra_slots = sorted(set(slots) - set(self.slot_definitions))
        errors.extend(f"UNKNOWN_SLOT:{slot}" for slot in extra_slots)

        for slot_name, value in slots.items():
            definition = self.slot_definitions.get(slot_name)
            if definition is None:
                continue
            normalized, error = self._validate_slot(
                slot_name,
                value,
                definition,
            )
            if error:
                errors.append(error)
            else:
                normalized_slots[slot_name] = normalized

        identity_error, identity_missing = self._validate_identity(
            intent,
            user_scope,
            normalized_slots,
        )
        if identity_error:
            errors.append(identity_error)
        if identity_missing:
            missing_slots.extend(identity_missing)

        for slot_name in intent["required_slots"]:
            if slot_name not in normalized_slots:
                missing_slots.append(slot_name)

        unique_missing = list(dict.fromkeys(missing_slots))
        allowed_tools = list(intent["allowed_tools"])
        if requested_tools is not None:
            errors.extend(
                self._validate_tool_plan(intent_id, requested_tools)
            )

        return ContractValidationResult(
            intent_id=intent_id,
            user_scope=user_scope,
            valid=not errors and not unique_missing,
            needs_clarification=bool(unique_missing),
            slots=normalized_slots,
            missing_slots=unique_missing,
            errors=errors,
            allowed_tools=allowed_tools,
        )

    def _validate_identity(
        self,
        intent: dict[str, Any],
        user_scope: str,
        slots: dict[str, Any],
    ) -> tuple[str | None, list[str]]:
        requirement = intent["identity_requirement"]
        if requirement == "none":
            return None, []
        if requirement == "customer_id":
            if "customer_id" not in slots:
                return "IDENTITY_REQUIRED:customer_id", ["customer_id"]
            return None, []
        if requirement == "dataset_id":
            if "dataset_id" not in slots:
                return "IDENTITY_REQUIRED:dataset_id", ["dataset_id"]
            return None, []
        if requirement == "customer_id_or_dataset_id":
            if user_scope == "existing_300":
                if "customer_id" not in slots:
                    return (
                        "IDENTITY_REQUIRED:customer_id",
                        ["customer_id"],
                    )
            elif user_scope == "cold_start":
                if "dataset_id" not in slots:
                    return (
                        "IDENTITY_REQUIRED:dataset_id",
                        ["dataset_id"],
                    )
            else:
                return "IDENTITY_NOT_AVAILABLE_FOR_SCOPE", ["customer_id"]
            return None, []
        return f"UNKNOWN_IDENTITY_REQUIREMENT:{requirement}", []

    def _validate_slot(
        self,
        slot_name: str,
        value: Any,
        definition: dict[str, Any],
    ) -> tuple[Any, str | None]:
        if value is None:
            return value, f"NULL_SLOT:{slot_name}"
        slot_type = definition["type"]

        if slot_type == "integer":
            if isinstance(value, bool):
                return value, f"INVALID_INTEGER:{slot_name}"
            try:
                normalized = int(value)
            except (TypeError, ValueError):
                return value, f"INVALID_INTEGER:{slot_name}"
            if "minimum" in definition and normalized < definition["minimum"]:
                return value, f"SLOT_BELOW_MINIMUM:{slot_name}"
            if "maximum" in definition and normalized > definition["maximum"]:
                return value, f"SLOT_ABOVE_MAXIMUM:{slot_name}"
            return normalized, None

        if slot_type == "string":
            if not isinstance(value, str) or not value.strip():
                return value, f"INVALID_STRING:{slot_name}"
            normalized = value.strip()
            if "enum" in definition and normalized not in definition["enum"]:
                return value, f"INVALID_ENUM:{slot_name}:{normalized}"
            if definition.get("format") == "date":
                try:
                    if len(normalized) != 10:
                        raise ValueError
                    date.fromisoformat(normalized)
                except ValueError:
                    return value, f"INVALID_DATE:{slot_name}"
            if definition.get("format") == "date-time":
                try:
                    parsed = datetime.fromisoformat(normalized)
                    if parsed.tzinfo is not None:
                        return value, f"TIMEZONE_NOT_ALLOWED:{slot_name}"
                except ValueError:
                    return value, f"INVALID_DATETIME:{slot_name}"
            return normalized, None

        if slot_type == "object":
            if not isinstance(value, dict):
                return value, f"INVALID_OBJECT:{slot_name}"
            return value, None

        return value, f"UNSUPPORTED_SLOT_TYPE:{slot_name}:{slot_type}"

    def _validate_tool_plan(
        self,
        intent_id: str,
        requested_tools: list[str],
    ) -> list[str]:
        errors: list[str] = []
        allowed = set(self.intents[intent_id]["allowed_tools"])
        for tool in requested_tools:
            if tool not in self.tools:
                errors.append(f"UNKNOWN_TOOL:{tool}")
            elif tool not in allowed:
                errors.append(f"TOOL_NOT_ALLOWED:{intent_id}:{tool}")
        if len(requested_tools) != len(set(requested_tools)):
            errors.append("DUPLICATE_TOOL_IN_PLAN")
        return errors
