from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    "useLimitSeconds": 1800,
    "nearLimitBeforeSeconds": 600,
    "awayThresholdSeconds": 600,
    "eventDebounceSeconds": 10,
    "personDetectionIntervalSeconds": 5,
    "tableDiffIntervalSeconds": 5,
    "tableChangeEnterThreshold": 0.18,
    "tableChangeExitThreshold": 0.10,
    "tableStaticThreshold": 0.012,
    "seatedPersonAnchorThreshold": 0.8,
    "identityChangeDistance": 0.35,
    "identityChangeConfirmSamples": 2,
    "embeddingWindowSize": 5,
    "identityEvidenceMaxPhotos": 5,
    "identityEvidenceDiversityDistance": 0.12,
}


@dataclass(frozen=True)
class SettingRule:
    kind: type
    min_value: float | None = None
    max_value: float | None = None


SETTING_RULES: dict[str, SettingRule] = {
    "useLimitSeconds": SettingRule(int, 60, 24 * 3600),
    "nearLimitBeforeSeconds": SettingRule(int, 0, 6 * 3600),
    "awayThresholdSeconds": SettingRule(int, 30, 12 * 3600),
    "eventDebounceSeconds": SettingRule(int, 0, 3600),
    "personDetectionIntervalSeconds": SettingRule(float, 1, 120),
    "tableDiffIntervalSeconds": SettingRule(float, 1, 600),
    "tableChangeEnterThreshold": SettingRule(float, 0, 1),
    "tableChangeExitThreshold": SettingRule(float, 0, 1),
    "tableStaticThreshold": SettingRule(float, 0, 1),
    "seatedPersonAnchorThreshold": SettingRule(float, 0, 1),
    "identityChangeDistance": SettingRule(float, 0, 2),
    "identityChangeConfirmSamples": SettingRule(int, 1, 20),
    "embeddingWindowSize": SettingRule(int, 1, 50),
    "identityEvidenceMaxPhotos": SettingRule(int, 1, 20),
    "identityEvidenceDiversityDistance": SettingRule(float, 0, 2),
}


class RuntimeSettings:
    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._values = dict(DEFAULT_SETTINGS)
        if initial:
            self.patch(initial)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def patch(self, updates: dict[str, Any]) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for key, value in updates.items():
            rule = SETTING_RULES.get(key)
            if rule is None:
                continue
            validated[key] = _coerce_and_validate(key, value, rule)

        with self._lock:
            next_values = dict(self._values)
            next_values.update(validated)
            if (
                next_values["tableChangeExitThreshold"]
                > next_values["tableChangeEnterThreshold"]
            ):
                raise ValueError("tableChangeExitThreshold must be <= tableChangeEnterThreshold")
            self._values = next_values
            return dict(self._values)


def _coerce_and_validate(key: str, value: Any, rule: SettingRule) -> Any:
    if rule.kind is str:
        coerced = str(value).strip()
        if not coerced:
            raise ValueError(f"{key} must not be empty")
        return coerced
    try:
        coerced = int(value) if rule.kind is int else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} has invalid value") from exc

    if rule.min_value is not None and coerced < rule.min_value:
        raise ValueError(f"{key} must be >= {rule.min_value}")
    if rule.max_value is not None and coerced > rule.max_value:
        raise ValueError(f"{key} must be <= {rule.max_value}")
    return coerced
