"""CARLA-independent lateral and longitudinal control intent over safe evidence.

This package turns the Mission 7 temporal lane estimate, the Mission 8
permission issued for that same evidence, and a detached signed observation of
the ego's own motion into bounded, immutable steering and speed *requests*, and
then bounds those two requests into one coherent command *envelope request*.  It
contains no actuator call, no CARLA type, and no vehicle command of any kind:
Mission 9 publishes a normalized steering intent in :mod:`control.lateral`,
Mission 10A publishes normalized throttle and brake intents in
:mod:`control.longitudinal`, Mission 10B publishes the combined, arbitrated,
rate-shaped envelope in :mod:`control.command`, and none of them is applied
here.  Any mapping onto a CARLA control belongs to a later mission and does not
exist.

Only the Mission 9 names are re-exported at package level.  The Mission 10A and
Mission 10B names are imported from :mod:`control.longitudinal` and
:mod:`control.command` directly and are deliberately *not* added to
``control.__all__``: an accepted Mission 9 test pins this package's top-level
public surface to carry no throttle, brake, speed, or acceleration vocabulary,
and neither Mission 10A nor Mission 10B modifies accepted tests.
"""

from control.lateral import (
    DEFAULT_LATERAL_CONTROLLER_CONFIG,
    MAX_LATERAL_GAIN,
    MAX_STEERING_RATE_PER_SECOND,
    LateralControlMode,
    LateralControlReason,
    LateralControlRequest,
    LateralController,
    LateralControllerConfig,
    LateralControllerMetrics,
)

__all__ = (
    "DEFAULT_LATERAL_CONTROLLER_CONFIG",
    "MAX_LATERAL_GAIN",
    "MAX_STEERING_RATE_PER_SECOND",
    "LateralControlMode",
    "LateralControlReason",
    "LateralControlRequest",
    "LateralController",
    "LateralControllerConfig",
    "LateralControllerMetrics",
)
