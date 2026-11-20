from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserScopeResult:
    user_scope: str
    reason_code: str
    customer_id: int | None
    dataset_id: str | None
    requires_upload_validation: bool


class UserScopeResolver:
    def __init__(
        self,
        *,
        known_customer_min: int = 1,
        known_customer_max: int = 300,
    ) -> None:
        if known_customer_min < 1 or known_customer_max < known_customer_min:
            raise ValueError("已有客户编号范围不合法")
        self.known_customer_min = known_customer_min
        self.known_customer_max = known_customer_max

    @classmethod
    def from_config(
        cls,
        path: str | Path | None = None,
    ) -> "UserScopeResolver":
        config_path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("assistant_config.toml")
        )
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        scope_config = config["user_scope"]
        return cls(
            known_customer_min=int(scope_config["known_customer_min"]),
            known_customer_max=int(scope_config["known_customer_max"]),
        )

    def resolve(
        self,
        *,
        customer_id: int | None,
        dataset_id: str | None,
        dataset_validated: bool = False,
    ) -> UserScopeResult:
        if customer_id is not None and (
            isinstance(customer_id, bool)
            or not isinstance(customer_id, int)
        ):
            raise ValueError("customer_id 必须是整数")
        normalized_dataset_id = (
            dataset_id.strip() if isinstance(dataset_id, str) else None
        )
        if normalized_dataset_id == "":
            normalized_dataset_id = None

        if (
            customer_id is not None
            and self.known_customer_min
            <= customer_id
            <= self.known_customer_max
        ):
            return UserScopeResult(
                user_scope="existing_300",
                reason_code="known_customer",
                customer_id=customer_id,
                dataset_id=normalized_dataset_id,
                requires_upload_validation=False,
            )

        if normalized_dataset_id is not None:
            return UserScopeResult(
                user_scope="cold_start" if dataset_validated else "unsupported",
                reason_code=(
                    "validated_cold_start_dataset"
                    if dataset_validated
                    else "upload_validation_required"
                ),
                customer_id=None,
                dataset_id=normalized_dataset_id,
                requires_upload_validation=not dataset_validated,
            )

        if customer_id is not None:
            return UserScopeResult(
                user_scope="unsupported",
                reason_code="unknown_customer_no_upload",
                customer_id=customer_id,
                dataset_id=None,
                requires_upload_validation=True,
            )

        return UserScopeResult(
            user_scope="unsupported",
            reason_code="missing_user_identity",
            customer_id=None,
            dataset_id=None,
            requires_upload_validation=True,
        )
