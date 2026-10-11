"""Unit tests for Mission 4F: SensorSuite integration into manual_drive.py.

Everything is deterministic: controlled clock values (plain floats), real
immutable telemetry records, and small recording fakes. No CARLA server, no
graphical display, and no timing sleeps are used.
"""

import dataclasses
import inspect
import unittest

import manual_drive
from manual_drive import (
    BRAKE_LEVEL,
    FPS,
    MAX_STEER,
    SNAPSHOT_INTERVAL_SECONDS,
    STEER_RATE,
    STEER_RETURN_RATE,
    THROTTLE_LEVEL,
    WARNING_SECONDS,
    WINDOW_SIZE,
    EventWarnings,
    IntervalGate,
    SensorHud,
    SessionState,
    attached_sensor_names,
    attachment_summary,
    cleanup_warning,
    extract_hud_state,
    perform_cleanup,
    run_with_cleanup,
    sensor_hud_lines,
    update_steer,
)
from sensors.suite import AttachReport, SensorAttachResult
from telemetry.freshness import (
    OperationalHealth,
    SensorFreshness,
    SensorKind,
    SensorLifecycle,
    SensorReadiness,
    SensorStatus,
)
from telemetry.state import (
    ActorRef,
    CameraFrame,
    CameraFrameMetadata,
    CollisionEvent,
    GnssState,
    ImuState,
    LaneInvasionEvent,
    LaneMarkingInfo,
    SampleStamp,
    TelemetrySnapshot,
    Vector3,
)

ALL_SENSOR_NAMES = ("gnss", "imu", "collision", "lane_invasion", "rgb_camera")


# -- builders ----------------------------------------------------------------


def stamp(frame=1, sim_time=1.0):
    return SampleStamp(frame=frame, sim_time=sim_time, monotonic=0.0)


def gnss_state():
    return GnssState(stamp=stamp(), latitude=1.0, longitude=2.0, altitude=3.0)


def imu_state():
    return ImuState(
        stamp=stamp(),
        accelerometer=Vector3(0.0, 0.0, 9.81),
        gyroscope=Vector3(0.0, 0.0, 0.0),
        compass_rad=0.0,
        heading_deg=0.0,
    )


def camera_frame():
    metadata = CameraFrameMetadata(
        stamp=stamp(),
        width_px=1,
        height_px=1,
        horizontal_fov_deg=90.0,
        pixel_format="RGB8",
        row_stride_bytes=3,
    )
    return CameraFrame(metadata=metadata, rgb_bytes=b"\x01\x02\x03")


def collision_event(other_type="vehicle.audi.tt", frame=1, sim_time=1.0):
    return CollisionEvent(
        stamp=SampleStamp(frame=frame, sim_time=sim_time, monotonic=0.0),
        self_actor=ActorRef(10, "vehicle.tesla.model3"),
        other_actor=ActorRef(99, other_type),
        normal_impulse=Vector3(1.0, 0.0, 0.0),
        impulse_magnitude=1.0,
    )


def lane_event(marking_types=("Solid", "Broken"), frame=1, sim_time=1.0):
    markings = tuple(
        LaneMarkingInfo(name, "White", "NONE", 0.15) for name in marking_types
    )
    return LaneInvasionEvent(
        stamp=SampleStamp(frame=frame, sim_time=sim_time, monotonic=0.0),
        crossed_markings=markings,
    )


def snap(**kwargs):
    return TelemetrySnapshot(revision=0, captured_monotonic=0.0, **kwargs)


def evaluated_status(
    name,
    kind=SensorKind.CONTINUOUS,
    *,
    lifecycle=SensorLifecycle.ATTACHED,
    readiness=None,
    freshness=None,
    health=OperationalHealth.HEALTHY,
    age=0.0,
    event_count=None,
    reason=None,
):
    if readiness is None:
        readiness = (
            SensorReadiness.READY
            if kind is SensorKind.CONTINUOUS
            else SensorReadiness.NOT_APPLICABLE
        )
    if freshness is None:
        freshness = (
            SensorFreshness.FRESH
            if kind is SensorKind.CONTINUOUS
            else SensorFreshness.NOT_APPLICABLE
        )
    return SensorStatus(
        name=name,
        kind=kind,
        lifecycle=lifecycle,
        readiness=readiness,
        freshness=freshness,
        operational_health=health,
        evaluated_monotonic=10.0,
        age_seconds=age,
        expected_interval=0.1 if kind is SensorKind.CONTINUOUS else None,
        stale_after=0.5 if kind is SensorKind.CONTINUOUS else None,
        accepted_count=0 if event_count is None else event_count,
        event_count=event_count,
        error_count=0,
        reason=reason,
    )


def report(unavailable=()):
    results = tuple(
        SensorAttachResult(name, name not in unavailable,
                           "boom" if name in unavailable else None)
        for name in ALL_SENSOR_NAMES
    )
    return AttachReport(results)


class CleanupBaseException(BaseException):
    pass


class RecordingSuite:
    def __init__(self, log, fail=False, failure=None):
        self._log = log
        self._failure = RuntimeError("suite boom") if fail else failure
        self.calls = 0

    def destroy(self):
        self.calls += 1
        self._log.append("suite")
        if self._failure is not None:
            raise self._failure


class RecordingVehicle:
    def __init__(self, log, fail=False, result=True, failure=None):
        self._log = log
        self._failure = RuntimeError("vehicle boom") if fail else failure
        self._result = result
        self.calls = 0

    def destroy(self):
        self.calls += 1
        self._log.append("vehicle")
        if self._failure is not None:
            raise self._failure
        return self._result


class RecordingDisplay:
    def __init__(self, log, failure=None):
        self._log = log
        self._failure = failure
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self._log.append("pygame")
        if self._failure is not None:
            raise self._failure


# -- attachment summary --------------------------------------------------------


class AttachmentSummaryTest(unittest.TestCase):
    def test_full_attachment(self):
        self.assertEqual(attachment_summary(report()), "Sensors: 5/5 attached")
        self.assertEqual(attached_sensor_names(report()), frozenset(ALL_SENSOR_NAMES))

    def test_degraded_attachment_names_missing_sensor(self):
        self.assertEqual(
            attachment_summary(report(unavailable=("rgb_camera",))),
            "Sensors: 4/5 attached; unavailable: rgb_camera",
        )

    def test_degraded_attachment_lists_all_missing(self):
        self.assertEqual(
            attachment_summary(report(unavailable=("gnss", "rgb_camera"))),
            "Sensors: 3/5 attached; unavailable: gnss, rgb_camera",
        )


# -- HUD state ------------------------------------------------------------------


class HudStateTest(unittest.TestCase):
    def test_empty_status_projection_is_unavailable(self):
        none_attached = attached_sensor_names(report(unavailable=ALL_SENSOR_NAMES))
        hud = extract_hud_state(snap(), none_attached)
        self.assertEqual(hud, SensorHud())
        lines = sensor_hud_lines(hud)
        self.assertIn("GNSS: UNAVAILABLE", lines[0])
        self.assertIn("COLLISION: UNAVAILABLE", lines[1])
        self.assertIn("LANE SENSOR: UNAVAILABLE", lines[1])

    def test_gnss_and_imu_ready_only_from_evaluated_fresh_status(self):
        snapshot = snap(
            gnss=gnss_state(),
            imu=imu_state(),
            statuses=(evaluated_status("gnss"), evaluated_status("imu")),
        )
        hud = extract_hud_state(snapshot, attached_sensor_names(report()))
        self.assertTrue(hud.gnss_ready)
        self.assertTrue(hud.imu_ready)
        self.assertFalse(hud.camera_ready)

        failed = dataclasses.replace(
            evaluated_status("gnss"),
            lifecycle=SensorLifecycle.ATTACHMENT_FAILED,
            readiness=SensorReadiness.NOT_APPLICABLE,
            freshness=SensorFreshness.NOT_APPLICABLE,
            operational_health=OperationalHealth.FAILED,
        )
        failed_snapshot = snap(gnss=gnss_state(), statuses=(failed,))
        self.assertFalse(
            extract_hud_state(
                failed_snapshot, attached_sensor_names(report())
            ).gnss_ready
        )

    def test_camera_waiting_then_fresh_from_status(self):
        attached = attached_sensor_names(report())
        waiting = evaluated_status(
            "rgb_camera",
            readiness=SensorReadiness.WAITING_FOR_FIRST_SAMPLE,
            freshness=SensorFreshness.WAITING_FOR_FIRST_SAMPLE,
        )
        self.assertFalse(
            extract_hud_state(snap(statuses=(waiting,)), attached).camera_ready
        )
        hud = extract_hud_state(
            snap(
                camera=camera_frame(),
                statuses=(evaluated_status("rgb_camera"),),
            ),
            attached,
        )
        self.assertTrue(hud.camera_ready)
        lines = sensor_hud_lines(hud)
        self.assertIn("CAMERA: FRESH", lines[0])

    def test_collision_and_lane_project_event_health(self):
        attached = attached_sensor_names(report(unavailable=("lane_invasion",)))
        collision = evaluated_status(
            "collision", SensorKind.EVENT, event_count=0
        )
        lane = evaluated_status(
            "lane_invasion",
            SensorKind.EVENT,
            lifecycle=SensorLifecycle.NOT_AVAILABLE,
            event_count=0,
        )
        hud = extract_hud_state(snap(statuses=(collision, lane)), attached)
        self.assertTrue(hud.collision_armed)      # armed without any event
        self.assertFalse(hud.lane_armed)
        lines = sensor_hud_lines(hud)
        self.assertIn("COLLISION: HEALTHY(0)", lines[1])
        self.assertIn("LANE SENSOR: UNAVAILABLE", lines[1])

    def test_hud_state_retains_no_camera_payload(self):
        hud = extract_hud_state(
            snap(
                camera=camera_frame(),
                statuses=(evaluated_status("rgb_camera"),),
            ),
            attached_sensor_names(report()),
        )
        for field in dataclasses.fields(SensorHud):
            value = getattr(hud, field.name)
            self.assertTrue(value is None or isinstance(value, SensorStatus))
            self.assertNotIsInstance(value, CameraFrame)

    def test_all_required_continuous_labels_and_bounded_ages(self):
        base = evaluated_status("gnss")
        cases = (
            (dataclasses.replace(base, readiness=SensorReadiness.WAITING_FOR_FIRST_SAMPLE, freshness=SensorFreshness.WAITING_FOR_FIRST_SAMPLE), "WAITING"),
            (base, "FRESH"),
            (dataclasses.replace(base, freshness=SensorFreshness.STALE, age_seconds=1.25), "STALE 1.2s"),
            (dataclasses.replace(base, operational_health=OperationalHealth.DEGRADED, reason="secret exception text"), "DEGRADED"),
            (dataclasses.replace(base, lifecycle=SensorLifecycle.ATTACHMENT_FAILED, operational_health=OperationalHealth.FAILED), "FAILED"),
            (dataclasses.replace(base, lifecycle=SensorLifecycle.NOT_AVAILABLE), "UNAVAILABLE"),
            (dataclasses.replace(base, lifecycle=SensorLifecycle.DESTROYED), "DESTROYED"),
        )
        for status, label in cases:
            with self.subTest(label=label):
                line = sensor_hud_lines(SensorHud(gnss=status))[0]
                self.assertIn(f"GNSS: {label}", line)
                self.assertNotIn("secret exception text", line)
                self.assertNotIn("READY", line)

        for age, expected in ((-1.0, "STALE 0.0s"), (1000.0, "STALE >999.9s"), (float("nan"), "STALE >999.9s"), (float("inf"), "STALE >999.9s")):
            with self.subTest(age=age):
                stale = dataclasses.replace(base, freshness=SensorFreshness.STALE, age_seconds=age)
                line = sensor_hud_lines(SensorHud(gnss=stale))[0]
                self.assertIn(expected, line)
                self.assertNotIn("nan", line.lower())
                self.assertNotIn("inf", line.lower())

    def test_all_required_event_labels_and_window_dimensions(self):
        base = evaluated_status("collision", SensorKind.EVENT, event_count=0)
        cases = (
            (base, "HEALTHY(0)"),
            (dataclasses.replace(base, operational_health=OperationalHealth.DEGRADED), "DEGRADED"),
            (dataclasses.replace(base, lifecycle=SensorLifecycle.ATTACHMENT_FAILED, operational_health=OperationalHealth.FAILED), "FAILED"),
            (dataclasses.replace(base, lifecycle=SensorLifecycle.NOT_AVAILABLE), "UNAVAILABLE"),
            (dataclasses.replace(base, lifecycle=SensorLifecycle.DESTROYED), "DESTROYED"),
        )
        for status, label in cases:
            with self.subTest(label=label):
                line = sensor_hud_lines(SensorHud(collision=status))[1]
                self.assertIn(f"COLLISION: {label}", line)
        self.assertEqual(WINDOW_SIZE, (460, 400))


# -- snapshot sampling gate -------------------------------------------------------


class IntervalGateTest(unittest.TestCase):
    def test_first_call_is_due_immediately(self):
        gate = IntervalGate(0.25)
        self.assertTrue(gate.due(100.0))
        self.assertFalse(gate.due(100.1))

    def test_rate_is_bounded_to_one_sample_per_interval(self):
        gate = IntervalGate(0.25)
        probes = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.49, 0.5, 0.6, 0.74, 0.75]
        fired = [now for now in probes if gate.due(now)]
        self.assertEqual(fired, [0.0, 0.25, 0.5, 0.75])   # one per interval

    def test_missed_intervals_are_skipped_without_catchup(self):
        gate = IntervalGate(0.25)
        self.assertTrue(gate.due(0.0))
        self.assertTrue(gate.due(1.1))     # long stall -> exactly one sample
        self.assertFalse(gate.due(1.2))    # no catch-up burst afterwards
        self.assertTrue(gate.due(1.25))    # grid preserved: no long-term drift

    def test_production_interval_is_ten_hz(self):
        self.assertEqual(SNAPSHOT_INTERVAL_SECONDS, 0.10)


# -- event warnings ----------------------------------------------------------------


class EventWarningsTest(unittest.TestCase):
    def test_new_collision_sequence_produces_one_warning(self):
        warnings = EventWarnings()
        warnings.observe(
            snap(collisions=(collision_event(),), collision_sequence=1), now=10.0
        )
        self.assertEqual(warnings.active(10.5), ["Collision: vehicle.audi.tt"])

    def test_duplicate_sequence_produces_no_new_warning(self):
        warnings = EventWarnings()
        snapshot = snap(collisions=(collision_event(),), collision_sequence=1)
        warnings.observe(snapshot, now=10.0)
        warnings.observe(snapshot, now=11.5)      # same sequence: no re-post
        self.assertEqual(len(warnings.active(11.9)), 1)
        self.assertEqual(warnings.active(12.0), [])   # expiry not extended

    def test_sequence_jump_shows_latest_event_with_count(self):
        warnings = EventWarnings()
        history = (
            collision_event(other_type="vehicle.audi.tt"),
            collision_event(other_type="walker.pedestrian.0001", frame=4),
        )
        warnings.observe(
            snap(collisions=history, collision_sequence=3), now=5.0
        )
        self.assertEqual(
            warnings.active(5.1), ["Collision: walker.pedestrian.0001 (3 new)"]
        )

    def test_lane_warning_uses_stable_marking_names(self):
        warnings = EventWarnings()
        warnings.observe(
            snap(lane_invasions=(lane_event(("Solid", "Broken")),),
                 lane_invasion_sequence=1),
            now=1.0,
        )
        self.assertEqual(warnings.active(1.5), ["Lane crossed: Solid, Broken"])

    def test_warning_expires_after_exactly_two_host_seconds(self):
        self.assertEqual(WARNING_SECONDS, 2.0)
        warnings = EventWarnings()
        warnings.observe(
            snap(collisions=(collision_event(),), collision_sequence=1), now=10.0
        )
        self.assertEqual(len(warnings.active(11.999)), 1)
        self.assertEqual(warnings.active(12.0), [])       # exactly 2.0 s later

    def test_simulation_time_rollback_does_not_affect_expiry(self):
        warnings = EventWarnings()
        warnings.observe(
            snap(collisions=(collision_event(sim_time=500.0),),
                 collision_sequence=1),
            now=10.0,
        )
        # A later snapshot whose simulation clock rolled far backwards (and no
        # new accepted events) must not disturb host-monotonic expiry.
        warnings.observe(
            snap(collisions=(collision_event(sim_time=1.0),),
                 collision_sequence=1),
            now=11.0,
        )
        self.assertEqual(len(warnings.active(11.9)), 1)
        self.assertEqual(warnings.active(12.0), [])

    def test_independent_collision_and_lane_tracking(self):
        warnings = EventWarnings()
        warnings.observe(
            snap(
                collisions=(collision_event(),),
                collision_sequence=1,
                lane_invasions=(lane_event(("Solid",)),),
                lane_invasion_sequence=1,
            ),
            now=0.0,
        )
        self.assertEqual(
            warnings.active(0.1),
            ["Collision: vehicle.audi.tt", "Lane crossed: Solid"],
        )


# -- cleanup orchestration -----------------------------------------------------------


class CleanupTest(unittest.TestCase):
    def assert_traceback_contains(self, traceback, code):
        while traceback is not None:
            if traceback.tb_frame.f_code is code:
                return
            traceback = traceback.tb_next
        self.fail(f"traceback did not contain {code.co_name}")

    def test_cleanup_order_suite_then_vehicle_then_display(self):
        log = []
        failures = perform_cleanup(
            RecordingSuite(log), RecordingVehicle(log), lambda: log.append("pygame")
        )
        self.assertEqual(log, ["suite", "vehicle", "pygame"])
        self.assertEqual(failures, [])
        self.assertIsNone(cleanup_warning(failures))

    def test_suite_failure_does_not_skip_vehicle_or_display(self):
        log = []
        failures = perform_cleanup(
            RecordingSuite(log, fail=True),
            RecordingVehicle(log),
            lambda: log.append("pygame"),
        )
        self.assertEqual(log, ["suite", "vehicle", "pygame"])
        self.assertEqual(len(failures), 1)

    def test_vehicle_failure_does_not_skip_display(self):
        log = []
        failures = perform_cleanup(
            RecordingSuite(log),
            RecordingVehicle(log, fail=True),
            lambda: log.append("pygame"),
        )
        self.assertEqual(log, ["suite", "vehicle", "pygame"])
        self.assertEqual(len(failures), 1)

    def test_all_failures_collected_into_one_warning(self):
        log = []

        def broken_quit():
            log.append("pygame")
            raise RuntimeError("display boom")

        failures = perform_cleanup(
            RecordingSuite(log, fail=True),
            RecordingVehicle(log, fail=True),
            broken_quit,
        )
        self.assertEqual(log, ["suite", "vehicle", "pygame"])  # all attempted
        self.assertEqual(len(failures), 3)
        warning = cleanup_warning(failures)
        self.assertTrue(warning.startswith("Cleanup warning: "))
        self.assertEqual(warning.count("Cleanup warning"), 1)  # consolidated

    def test_partial_initialization_with_no_suite(self):
        log = []
        failures = perform_cleanup(
            None, RecordingVehicle(log), lambda: log.append("pygame")
        )
        self.assertEqual(log, ["vehicle", "pygame"])
        self.assertEqual(failures, [])

    def test_vehicle_destroy_false_is_recorded(self):
        log = []
        failures = perform_cleanup(
            None, RecordingVehicle(log, result=False), lambda: log.append("pygame")
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("vehicle", failures[0])

    def test_partial_initialization_matrix_calls_each_present_stage_once(self):
        cases = (
            ("all", True, True, ["suite", "vehicle", "pygame"]),
            ("no suite", False, True, ["vehicle", "pygame"]),
            ("no vehicle", True, False, ["suite", "pygame"]),
            ("display only", False, False, ["pygame"]),
        )
        for name, has_suite, has_vehicle, expected in cases:
            with self.subTest(name=name):
                log = []
                suite = RecordingSuite(log) if has_suite else None
                vehicle = RecordingVehicle(log) if has_vehicle else None
                display = RecordingDisplay(log)
                self.assertEqual(perform_cleanup(suite, vehicle, display), [])
                self.assertEqual(log, expected)
                if suite is not None:
                    self.assertEqual(suite.calls, 1)
                if vehicle is not None:
                    self.assertEqual(vehicle.calls, 1)
                self.assertEqual(display.calls, 1)

    def test_each_ordinary_exception_preserves_order_and_diagnostic(self):
        for stage in ("sensor suite", "vehicle", "display"):
            with self.subTest(stage=stage):
                log = []
                failure = RuntimeError(f"{stage} boom")
                suite = RecordingSuite(
                    log, failure=failure if stage == "sensor suite" else None
                )
                vehicle = RecordingVehicle(
                    log, failure=failure if stage == "vehicle" else None
                )
                display = RecordingDisplay(
                    log, failure=failure if stage == "display" else None
                )
                failures = perform_cleanup(suite, vehicle, display)
                self.assertEqual(log, ["suite", "vehicle", "pygame"])
                self.assertEqual(suite.calls, 1)
                self.assertEqual(vehicle.calls, 1)
                self.assertEqual(display.calls, 1)
                self.assertEqual(
                    failures, [f"{stage}: RuntimeError('{stage} boom')"]
                )

    def test_each_stage_defers_each_nonordinary_base_exception(self):
        exception_factories = (
            ("KeyboardInterrupt", lambda: KeyboardInterrupt("stop")),
            ("SystemExit", lambda: SystemExit(19)),
            ("custom", lambda: CleanupBaseException("custom stop")),
        )
        stage_codes = {
            "sensor suite": RecordingSuite.destroy.__code__,
            "vehicle": RecordingVehicle.destroy.__code__,
            "display": RecordingDisplay.__call__.__code__,
        }
        for stage in ("sensor suite", "vehicle", "display"):
            for kind, exception_factory in exception_factories:
                with self.subTest(stage=stage, kind=kind):
                    log = []
                    failure = exception_factory()
                    suite = RecordingSuite(
                        log, failure=failure if stage == "sensor suite" else None
                    )
                    vehicle = RecordingVehicle(
                        log, failure=failure if stage == "vehicle" else None
                    )
                    display = RecordingDisplay(
                        log, failure=failure if stage == "display" else None
                    )
                    try:
                        perform_cleanup(suite, vehicle, display)
                    except BaseException as caught:
                        caught_exception = caught
                        caught_traceback = caught.__traceback__
                    else:
                        self.fail("cleanup BaseException was not re-raised")
                    self.assertIs(caught_exception, failure)
                    self.assertEqual(log, ["suite", "vehicle", "pygame"])
                    self.assertEqual(suite.calls, 1)
                    self.assertEqual(vehicle.calls, 1)
                    self.assertEqual(display.calls, 1)
                    self.assert_traceback_contains(caught_traceback, stage_codes[stage])

    def test_multiple_cleanup_failures_have_deterministic_precedence(self):
        cases = (
            (
                "suite base then success",
                CleanupBaseException("suite base"), None, None, True, "suite",
            ),
            (
                "suite base then vehicle exception",
                KeyboardInterrupt("suite interrupt"),
                RuntimeError("vehicle ordinary"), None, True, "suite",
            ),
            (
                "suite base vehicle base display exception",
                SystemExit(21), CleanupBaseException("vehicle base"),
                RuntimeError("display ordinary"), True, "suite",
            ),
            (
                "suite exception vehicle base display base",
                RuntimeError("suite ordinary"), SystemExit(22),
                CleanupBaseException("display base"), True, "vehicle",
            ),
            (
                "all ordinary",
                RuntimeError("suite ordinary"), ValueError("vehicle ordinary"),
                OSError("display ordinary"), True, None,
            ),
            (
                "all base",
                KeyboardInterrupt("suite base"), SystemExit(23),
                CleanupBaseException("display base"), True, "suite",
            ),
            (
                "false vehicle plus display base",
                None, None, CleanupBaseException("display base"), False, "display",
            ),
        )
        for name, suite_error, vehicle_error, display_error, result, winner in cases:
            with self.subTest(name=name):
                log = []
                suite = RecordingSuite(log, failure=suite_error)
                vehicle = RecordingVehicle(
                    log, result=result, failure=vehicle_error
                )
                display = RecordingDisplay(log, failure=display_error)
                expected_failures = []
                if suite_error is not None:
                    expected_failures.append(f"sensor suite: {suite_error!r}")
                if vehicle_error is not None:
                    expected_failures.append(f"vehicle: {vehicle_error!r}")
                elif not result:
                    expected_failures.append("vehicle: destroy() returned False")
                if display_error is not None:
                    expected_failures.append(f"display: {display_error!r}")

                if winner is None:
                    self.assertEqual(
                        perform_cleanup(suite, vehicle, display), expected_failures
                    )
                else:
                    winning_error = {
                        "suite": suite_error,
                        "vehicle": vehicle_error,
                        "display": display_error,
                    }[winner]
                    with self.assertRaises(BaseException) as context:
                        perform_cleanup(suite, vehicle, display)
                    self.assertIs(context.exception, winning_error)
                    self.assertEqual(
                        context.exception.__notes__,
                        [cleanup_warning(expected_failures)],
                    )
                self.assertEqual(log, ["suite", "vehicle", "pygame"])
                self.assertEqual((suite.calls, vehicle.calls, display.calls), (1, 1, 1))

    def test_unprintable_exception_diagnostic_cannot_skip_later_stages(self):
        class UnprintableError(RuntimeError):
            def __repr__(self):
                raise CleanupBaseException("repr failed")

        log = []
        failures = perform_cleanup(
            RecordingSuite(log, failure=UnprintableError("suite")),
            RecordingVehicle(log),
            RecordingDisplay(log),
        )
        self.assertEqual(log, ["suite", "vehicle", "pygame"])
        self.assertEqual(
            failures,
            ["sensor suite: <UnprintableError repr failed: CleanupBaseException>"],
        )
        self.assertLessEqual(len(cleanup_warning(failures)), 300)


class RunWithCleanupTest(unittest.TestCase):
    def _state(self, log):
        state = SessionState()
        state.suite = RecordingSuite(log)
        state.vehicle = RecordingVehicle(log)
        return state

    def test_normal_exit_reaches_cleanup(self):
        # Esc and window-close both make the drive loop (the action) return
        # normally; cleanup must still run in full order.
        log, lines = [], []
        code = run_with_cleanup(
            lambda: None, self._state(log), lambda: log.append("pygame"), lines.append
        )
        self.assertEqual(code, 0)
        self.assertEqual(log, ["suite", "vehicle", "pygame"])

    def test_keyboard_interrupt_reaches_cleanup(self):
        log, lines = [], []

        def interrupted():
            raise KeyboardInterrupt

        code = run_with_cleanup(
            interrupted, self._state(log), lambda: log.append("pygame"), lines.append
        )
        self.assertEqual(code, 0)
        self.assertEqual(log, ["suite", "vehicle", "pygame"])
        self.assertTrue(any("Interrupted" in line for line in lines))

    def test_disconnect_reaches_cleanup_with_one_error_line(self):
        log, lines = [], []

        def disconnected():
            raise RuntimeError("time-out of 10000ms while waiting for the simulator")

        code = run_with_cleanup(
            disconnected, self._state(log), lambda: log.append("pygame"), lines.append
        )
        self.assertEqual(code, 1)
        self.assertEqual(log, ["suite", "vehicle", "pygame"])
        error_lines = [line for line in lines if line.startswith("ERROR")]
        self.assertEqual(len(error_lines), 1)

    def test_programming_error_still_cleans_up_then_propagates(self):
        log, lines = [], []

        def buggy():
            raise ValueError("programming error")

        with self.assertRaises(ValueError):
            run_with_cleanup(
                buggy, self._state(log), lambda: log.append("pygame"), lines.append
            )
        self.assertEqual(log, ["suite", "vehicle", "pygame"])

    def test_cleanup_failures_produce_single_consolidated_warning(self):
        log, lines = [], []
        state = SessionState()
        state.suite = RecordingSuite(log, fail=True)
        state.vehicle = RecordingVehicle(log, fail=True)
        code = run_with_cleanup(
            lambda: None, state, lambda: log.append("pygame"), lines.append
        )
        self.assertEqual(code, 0)
        warnings = [line for line in lines if line.startswith("Cleanup warning")]
        self.assertEqual(len(warnings), 1)

    def test_cleanup_base_exception_after_success_is_deferred_and_reraised(self):
        log, lines = [], []
        failure = KeyboardInterrupt("cleanup interrupt")
        state = SessionState(
            suite=RecordingSuite(log, failure=failure),
            vehicle=RecordingVehicle(log),
        )
        display = RecordingDisplay(log)
        with self.assertRaises(KeyboardInterrupt) as context:
            run_with_cleanup(lambda: None, state, display, lines.append)
        self.assertIs(context.exception, failure)
        self.assertEqual(log, ["suite", "vehicle", "pygame"])
        self.assertEqual(
            (state.suite.calls, state.vehicle.calls, display.calls), (1, 1, 1)
        )
        self.assertEqual(
            [line for line in lines if line.startswith("Cleanup warning")],
            ["Cleanup warning: sensor suite: KeyboardInterrupt('cleanup interrupt')"],
        )

    def test_translated_action_failures_keep_exit_codes_over_cleanup_base(self):
        cases = (
            (
                "keyboard interrupt", KeyboardInterrupt("action interrupt"), 0,
                "Interrupted", CleanupBaseException("cleanup base"),
            ),
            (
                "runtime error", RuntimeError("simulator disconnected"), 1,
                "ERROR", SystemExit(44),
            ),
        )
        for name, action_error, expected_code, action_text, cleanup_error in cases:
            with self.subTest(name=name):
                log, lines = [], []
                state = SessionState(
                    suite=RecordingSuite(log, failure=cleanup_error),
                    vehicle=RecordingVehicle(log),
                )
                display = RecordingDisplay(log)

                def action():
                    raise action_error

                code = run_with_cleanup(action, state, display, lines.append)
                self.assertEqual(code, expected_code)
                self.assertEqual(log, ["suite", "vehicle", "pygame"])
                self.assertEqual(
                    (state.suite.calls, state.vehicle.calls, display.calls),
                    (1, 1, 1),
                )
                self.assertTrue(any(action_text in line for line in lines))
                self.assertEqual(
                    len([line for line in lines if line.startswith("Cleanup warning")]),
                    1,
                )

    def test_propagated_action_failures_outrank_cleanup_failures(self):
        cases = (
            (
                "value error plus suite interrupt",
                ValueError("action value"),
                KeyboardInterrupt("suite cleanup"), None, None,
            ),
            (
                "system exit plus every cleanup failure",
                SystemExit(37),
                RuntimeError("suite cleanup"),
                CleanupBaseException("vehicle cleanup"),
                SystemExit(99),
            ),
            (
                "custom base plus vehicle exit",
                CleanupBaseException("action base"),
                None, SystemExit(38), RuntimeError("display cleanup"),
            ),
        )
        for name, action_error, suite_error, vehicle_error, display_error in cases:
            with self.subTest(name=name):
                log, lines = [], []
                state = SessionState(
                    suite=RecordingSuite(log, failure=suite_error),
                    vehicle=RecordingVehicle(log, failure=vehicle_error),
                )
                display = RecordingDisplay(log, failure=display_error)

                def action():
                    raise action_error

                with self.assertRaises(BaseException) as context:
                    run_with_cleanup(action, state, display, lines.append)
                self.assertIs(context.exception, action_error)
                if isinstance(action_error, SystemExit):
                    self.assertEqual(context.exception.code, 37)
                self.assertEqual(log, ["suite", "vehicle", "pygame"])
                self.assertEqual(
                    (state.suite.calls, state.vehicle.calls, display.calls),
                    (1, 1, 1),
                )
                self.assertEqual(
                    len([line for line in lines if line.startswith("Cleanup warning")]),
                    1,
                )
                self.assertEqual(
                    action_error.__notes__,
                    [line for line in lines if line.startswith("Cleanup warning")],
                )

    def test_false_vehicle_result_keeps_success_code_and_warning_behavior(self):
        log, lines = [], []
        vehicle = RecordingVehicle(log, result=False)
        display = RecordingDisplay(log)
        code = run_with_cleanup(
            lambda: None,
            SessionState(vehicle=vehicle),
            display,
            lines.append,
        )
        self.assertEqual(code, 0)
        self.assertEqual(log, ["vehicle", "pygame"])
        self.assertEqual((vehicle.calls, display.calls), (1, 1))
        self.assertEqual(
            lines, ["Cleanup warning: vehicle: destroy() returned False"]
        )

    def test_display_only_partial_initialization_runs_once(self):
        log, lines = [], []
        display = RecordingDisplay(log)
        code = run_with_cleanup(
            lambda: None, SessionState(), display, lines.append
        )
        self.assertEqual(code, 0)
        self.assertEqual(log, ["pygame"])
        self.assertEqual(display.calls, 1)
        self.assertEqual(lines, [])


# -- controls must remain unchanged ---------------------------------------------------


class ControlsUnchangedTest(unittest.TestCase):
    def test_control_constants_unchanged(self):
        self.assertEqual(FPS, 60)
        self.assertEqual(THROTTLE_LEVEL, 0.7)
        self.assertEqual(BRAKE_LEVEL, 1.0)
        self.assertEqual(MAX_STEER, 0.7)
        self.assertEqual(STEER_RATE, 1.2)
        self.assertEqual(STEER_RETURN_RATE, 2.0)

    def test_steering_ramp_and_centering_unchanged(self):
        self.assertAlmostEqual(update_steer(0.0, 1, 0.1), 0.12)     # ramp up
        self.assertAlmostEqual(update_steer(0.0, -1, 0.1), -0.12)   # ramp left
        self.assertAlmostEqual(update_steer(0.5, 0, 0.1), 0.30)     # re-center
        self.assertAlmostEqual(update_steer(-0.5, 0, 0.1), -0.30)
        self.assertAlmostEqual(update_steer(0.69, 1, 1.0), MAX_STEER)  # clamp

    def test_key_mappings_still_present_in_drive_loop(self):
        source = inspect.getsource(manual_drive.drive_loop)
        for token in (
            "K_w", "K_UP", "K_s", "K_DOWN", "K_a", "K_LEFT", "K_d", "K_RIGHT",
            "K_SPACE", "K_r", "K_ESCAPE", "pygame.QUIT",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
