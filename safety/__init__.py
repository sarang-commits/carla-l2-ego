"""CARLA-independent safety-permission supervision for perception evidence.

This package decides whether autonomous control may currently be permitted.
It contains no steering, throttle, braking, or other vehicle command.
"""

from safety.supervisor import (
    DEFAULT_SAFETY_SUPERVISOR_CONFIG,
    MAX_REQUIRED_HEALTHY_RESULTS,
    MAX_SAFETY_RESULT_AGE_SECONDS,
    ComponentHealth,
    SafetyDecision,
    SafetyReason,
    SafetyState,
    SafetySupervisor,
    SafetySupervisorConfig,
    SafetySupervisorMetrics,
)

__all__ = (
    "DEFAULT_SAFETY_SUPERVISOR_CONFIG",
    "MAX_REQUIRED_HEALTHY_RESULTS",
    "MAX_SAFETY_RESULT_AGE_SECONDS",
    "ComponentHealth",
    "SafetyDecision",
    "SafetyReason",
    "SafetyState",
    "SafetySupervisor",
    "SafetySupervisorConfig",
    "SafetySupervisorMetrics",
)
