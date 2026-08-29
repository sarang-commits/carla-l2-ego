# carla-l2-exhibition

Development project for CARLA 0.9.16 on Windows. Current scope: environment
setup, Python-to-CARLA connectivity verification, ego-vehicle lifecycle,
manual keyboard driving with a five-sensor telemetry HUD, offline/live
lifecycle and memory-stability validation, single-frame camera lane perception,
a bounded perception runtime with live camera integration, bounded temporal
lane tracking, bounded safety supervision, and a bounded lateral steering
intent producer.

Mission 6B live perception integration is the accepted live baseline. Temporal
tracking (Mission 7), safety supervision (Mission 8), and lateral control
(Mission 9) are offline-accepted only, and Missions 10-19 remain unimplemented.

Despite the repository name, this project does not implement autopilot,
autonomous driving, or Level 2 driving assistance. The only code that actuates a
vehicle is the human-driven keyboard path in `manual_drive.py`. Nothing derived
from perception reaches a vehicle: the Mission 5-9 stack ends at a lateral
steering *request*, and no code turns that request into a vehicle command.

## Requirements

| Component | Version |
|-----------|---------|
| Python    | 3.12.10 (CPython, 64-bit) |
| CARLA     | 0.9.16 (installed at `C:\Users\kksre\Downloads\CARLA_Latest`) |

## Setup

### 1. Activate the virtual environment

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in cmd.exe:

```bat
.venv\Scripts\activate.bat
```

### 2. Install dependencies

The CARLA client is installed from the wheel bundled with the simulator
(guaranteed to match the server version):

```powershell
python -m pip install --upgrade pip
python -m pip install "C:\Users\kksre\Downloads\CARLA_Latest\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"
python -m pip install -r requirements.txt
```

## Launch the CARLA simulator

Start the simulator manually (it listens on port 2000 by default):

```powershell
C:\Users\kksre\Downloads\CARLA_Latest\CarlaUE4.exe
```

Wait until the 3D world is fully loaded before running any client script.

## Run the smoke test

With the simulator running and the virtual environment activated:

```powershell
python smoke_test.py
```

### Expected successful output

```text
CARLA client version: 0.9.16
Connecting to localhost:2000 (timeout: 10s)...
CARLA server version: 0.9.16
Map name:             Carla/Maps/Town10HD_Opt
Actor count:          <some number, e.g. 1-50>
Smoke test PASSED.
```

(The map name and actor count depend on what the server currently has
loaded; `Town10HD_Opt` is the default map.)

If the simulator is not running, the script exits with an error message
pointing to the manual launch path above.

## Mission 2: spawn and remove one ego vehicle

Spawns a single ego vehicle (preferring `vehicle.tesla.model3`, with a safe
fallback to another vehicle blueprint), moves the spectator camera above and
behind it, keeps it alive for 15 seconds, then always destroys it — including
on errors or Ctrl+C — via a `finally` block. No autopilot, sensors, or
controls are used.

With the simulator running and the virtual environment activated, from the
project root:

```powershell
python spawn_ego_vehicle.py
```

Or without activating the environment:

```powershell
.\.venv\Scripts\python.exe spawn_ego_vehicle.py
```

### Expected successful output

```text
Vehicle ID:     <some integer, e.g. 52>
Blueprint:      vehicle.tesla.model3
Spawn location: x=<...>, y=<...>, z=<...>
Spectator moved above and behind the vehicle.
Keeping vehicle alive for 15 seconds...
Cleanup: vehicle <id> destroyed.
```

Watch the simulator window: the camera jumps to the newly spawned vehicle,
which disappears again after 15 seconds. If the server is not running, the
script exits with a connection error and the manual launch path.

## Manual keyboard driving with sensor telemetry

Spawns one ego vehicle (preferring `vehicle.tesla.model3`, with a safe
fallback) and lets you drive it with the keyboard. A small pygame window
(460x400) shows live speed, throttle, brake, steering, reverse/handbrake
state, sensor readiness, collision/lane-crossing warnings, and the control
instructions. The spectator camera follows above and behind the vehicle. The
control loop runs at 60 FPS; steering ramps up gradually while a key is held
and returns toward center when released.

Every exit uses the same ordered teardown: `SensorSuite` -> ego vehicle ->
display/pygame. Each initialized stage is attempted exactly once, and later
stages still run when an earlier stage raises either `Exception` or another
`BaseException`. Ordinary cleanup failures are reported in one consolidated
warning. If the driving action already has an exception to propagate, that
same exception remains primary; otherwise, the first cleanup `BaseException`
is re-raised after all teardown stages finish. Ctrl+C retains the established
manual-drive behavior: it prints an interruption message, completes teardown,
and returns success. A CARLA-style `RuntimeError` prints one error line,
completes teardown, and returns exit code 1.

With the simulator running and the virtual environment activated, from the
project root:

```powershell
python manual_drive.py
```

Or without activating the environment:

```powershell
.\.venv\Scripts\python.exe manual_drive.py
```

The pygame window must have focus for the keys to register (click it after
it opens).

### Controls

| Key                 | Action                        |
|---------------------|-------------------------------|
| `W` / `Up Arrow`    | Throttle                      |
| `S` / `Down Arrow`  | Brake                         |
| `A` / `Left Arrow`  | Steer left (gradual)          |
| `D` / `Right Arrow` | Steer right (gradual)         |
| `Space`             | Handbrake (hold)              |
| `R`                 | Toggle reverse                |
| `Esc` / close window| Quit (vehicle is cleaned up)  |

### Sensor status semantics

The HUD keeps three independent dimensions instead of treating every sensor as
simply ready or not ready. Readiness says whether a required continuous sensor
has produced its first accepted sample. Freshness says whether that accepted
sample is still recent enough at the current host-monotonic evaluation time.
Operational health records active callback, source-order, or clock degradation;
its cumulative error history remains available after active health recovers.

GNSS, IMU, and the RGB camera are continuous sensors. Their configurable,
conservative defaults are:

| Sensor | Expected interval | Stale after | Startup grace |
|--------|-------------------|-------------|---------------|
| GNSS | 0.10 s | 0.50 s | 1.00 s |
| IMU | 0.05 s | 0.25 s | 0.50 s |
| RGB camera | 0.10 s | 0.50 s | 1.00 s |

An age exactly equal to `stale after` remains fresh; only a greater age is
stale. The same equality-pass rule applies to startup grace. Old, duplicate,
out-of-order, or otherwise rejected callbacks do not refresh age. One later
accepted valid sample immediately restores freshness and clears active
degradation while retaining the cumulative error count and bounded historical
diagnostic.

Collision and lane-invasion sensors are event-driven. They have no startup
sample requirement and no stale timeout: a correctly attached sensor with zero
events remains `HEALTHY(0)` for any duration. Event silence is not evidence of
failure. An accepted event updates its bounded history/count and can clear an
active callback error.

Every suite snapshot reads one injected host-monotonic clock value and evaluates
all five immutable statuses at that same effective time. CARLA frame and
simulation timestamps remain source-order diagnostics and are never mixed into
host-age calculations. Host clock rollback is clamped so age cannot become
negative, reported as bounded degradation, and cleared after coherent
nondecreasing progression resumes.

The two sensor HUD lines use only bounded labels: continuous sensors show
`WAITING`, `FRESH`, `STALE <age>`, `DEGRADED`, `FAILED`, `UNAVAILABLE`, or
`DESTROYED`; event sensors show `HEALTHY(<count>)` or the applicable lifecycle/
health label. Raw exception text is never rendered. These classifications are
telemetry only: they cause no autonomous control, failover, braking, steering,
or other vehicle response. Mission 4G readiness, report, verdict, live-cycle,
and acceptance behavior are unchanged. The thresholds above are configurable
defaults, not measured CARLA timing guarantees.

### Expected successful output

```text
Vehicle ID:     <some integer>
Blueprint:      vehicle.tesla.model3
Spawn location: x=<...>, y=<...>, z=<...>
Sensors: 5/5 attached
Driving started. Press Esc (or close the window) to quit.
Cleanup complete.
```

## Mission 4G.5: cycle-count and timed sensor suite validation

The live integration test requires a compatible CARLA server to be explicitly
started and fully loaded. It is skipped unless `CARLA_INTEGRATION` is exactly
`1`. With the virtual environment activated, run:

Cycle-count mode is selected when `CARLA_INTEGRATION_SOAK_SECONDS` is unset or
numeric zero. It executes exactly 1-20 complete cycles unless fail-fast stops
the run:

```powershell
$env:CARLA_INTEGRATION = "1"
$env:CARLA_INTEGRATION_CYCLES = "3"  # Optional: 1-20; defaults to 1
$env:CARLA_INTEGRATION_SOAK_SECONDS = "0"
python -m unittest tests.integration_sensor_suite
```

Timed-soak mode is selected by a positive soak duration from 120 through 3600
seconds. Leave `CARLA_INTEGRATION_CYCLES` unset in this mode; supplying explicit
cycles together with a positive soak is rejected before any live work:

```powershell
$env:CARLA_INTEGRATION = "1"
Remove-Item Env:CARLA_INTEGRATION_CYCLES -ErrorAction SilentlyContinue
$env:CARLA_INTEGRATION_SOAK_SECONDS = "120"
python -m unittest tests.integration_sensor_suite
```

Each cycle reconnects to CARLA and performs a fresh complete lifecycle: version
and world-policy checks, ego spawn, five sensor attachments, telemetry
validation, cleanup, and disappearance confirmation. Cycles run back-to-back,
without pacing or sleeping. A later cycle starts only after all owned actors
from the current cycle are confirmed absent. The run fails fast, so no later
cycle starts after any functional, cleanup, or disappearance failure.

Disappearance is confirmed only through fresh per-frame world snapshots
(`WorldSnapshot.has_actor`) under one bounded monotonic deadline, progressing
the simulation solely through the mode's authorized mechanism (`wait_for_tick`,
or `tick` when this harness owns ticking). The client-cached `world.get_actor`
lookup can keep returning actors this client spawned even after their
successful destruction, so it is never used as disappearance evidence.

In timed-soak mode, the requested duration is a cycle-admission window rather
than a hard cutoff. A cycle that begins before the host-monotonic deadline
receives its full existing spawn, readiness, cleanup, and disappearance
deadlines and may finish after the requested soak duration. No new cycle starts
at or after the total deadline.

Timed execution retains bounded aggregate evidence: every summary through 32
cycles, otherwise the first 8 and rolling final 24 summaries, plus exact total
counters and omission counts. Cleanup diagnostics are similarly capped at 8.
The separately authorized live acceptance run for timed soak uses the minimum
120-second duration; ordinary implementation tests remain offline and do not
connect to CARLA.

This validation does not perform autonomous driving, autopilot, or vehicle
control.

## Mission 4G.6: memory stability during timed soak

Timed-soak runs can additionally monitor the CARLA server process memory and
produce a memory-stability verdict. The monitor is bounded, runs only around
the timed-soak outer runner, and leaves the single-cycle runner memory-unaware.

Memory stability is opt-in and is accepted only when all of the following
hold; any other combination is rejected before any live work:

- `CARLA_INTEGRATION_MEMORY_STABILITY` is exactly `1` (absent or exactly `0`
  disables it; anything else is invalid);
- timed-soak mode is selected: `CARLA_INTEGRATION_SOAK_SECONDS` is positive
  and `CARLA_INTEGRATION_CYCLES` is unset;
- the raw, unmodified `CARLA_HOST` value is exactly `localhost`, `127.0.0.1`,
  or `::1` (the sampled process must be the local server). The memory
  contract performs no whitespace stripping and no case folding: values such
  as `" localhost"`, `"localhost "`, or `"Localhost"` are rejected even
  though legacy host parsing outside this contract still strips whitespace.
  An unset `CARLA_HOST` uses the default `localhost` and is accepted.

All six thresholds are required when enabled (strict ASCII parsing, no
defaults; byte limits are integers 0..2^63-1, percents are finite values
0..10000):

```powershell
$env:CARLA_INTEGRATION = "1"
Remove-Item Env:CARLA_INTEGRATION_CYCLES -ErrorAction SilentlyContinue
$env:CARLA_INTEGRATION_SOAK_SECONDS = "120"
$env:CARLA_INTEGRATION_MEMORY_STABILITY = "1"
$env:CARLA_MEMORY_WORKING_SET_MAX_PEAK_GROWTH_BYTES = "1073741824"
$env:CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_BYTES = "536870912"
$env:CARLA_MEMORY_WORKING_SET_MAX_FINAL_GROWTH_PERCENT = "25"
$env:CARLA_MEMORY_PRIVATE_COMMIT_MAX_PEAK_GROWTH_BYTES = "1073741824"
$env:CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_BYTES = "536870912"
$env:CARLA_MEMORY_PRIVATE_COMMIT_MAX_FINAL_GROWTH_PERCENT = "25"
python -m unittest tests.integration_sensor_suite
```

Optional variables: `CARLA_MEMORY_WORKING_SET_MAX_SLOPE_BYTES_PER_MINUTE` and
`CARLA_MEMORY_PRIVATE_COMMIT_MAX_SLOPE_BYTES_PER_MINUTE` (finite,
non-negative OLS slope limits), and `CARLA_SERVER_PID` (an explicit PID
1..4294967295 selecting one `CarlaUE4-Win64-Shipping.exe` process when
several are running).

The sampler (`tests/windows_memory_sampler.py`) is stdlib-`ctypes`-only. It
resolves exactly one `CarlaUE4-Win64-Shipping.exe` process through Toolhelp,
opens and retains one handle (so a recycled PID can never impersonate the
original process, and no replacement process is ever followed), records the
PID, image path, and creation time, reads `PROCESS_MEMORY_COUNTERS_EX`
(`WorkingSetSize` and `PrivateUsage`; `PagefileUsage` is only an
alias-consistency diagnostic), and detects process exit with a zero-timeout
`WaitForSingleObject`. GPU dedicated memory is sampled through PDH
`GPU Process Memory` instances beginning with `pid_<pid>_`; GPU evidence is
advisory only and never gates the verdict. The frozen executable evidence
must be a valid absolute Windows path whose normalized basename is exactly
`CarlaUE4-Win64-Shipping.exe`; empty, overlength, control-character, NUL, or
path/basename-mismatched evidence cannot support `STABLE`.

Sampling runs on a background thread with an exact 5-second cadence on
absolute monotonic deadlines and an interruptible event wait: one baseline
sample before the first cycle admission, at most one sample per wake (late
wakes never burst or backfill; skipped intervals after a clock jump are
counted in constant control flow from the exact rational values of the input
floats, never by ordinary float division or looping per interval),
sampling continues through the final cycle's cleanup and disappearance
verification, and one final sample follows the outer loop (coalesced when
its timestamp equals the previous sample's). The stop event is checked after
each wake and clock read, after deadline advancement, and immediately before
every periodic native sample. Retained timestamps must be finite,
non-negative, and strictly increasing. Retention is bounded to 256
samples (first 16 plus the rolling final 240) with exact
total/retained/omitted/missed counts and at most 8 sampling diagnostics;
metrics and the constant-space OLS slope still include every valid sample.
The OLS slope accumulates exact integer byte deltas against the first
sample's byte origin, so absolute counters beyond the float exact-integer
range (2^53) never erase small growth.

Every native operation — process resolution and open, each counter read,
each advisory GPU read, and every handle close — runs under a deterministic
bounded completion contract: the controlling thread waits at most 5 seconds
per native call and at most 20 seconds for the sampling thread to join. A
native call that never returns is abandoned on its daemon worker (the test
process always regains control and can exit), the failed call is never
retried, no replacement PID is acquired, and the affected handle is never
closed from another execution context; the run is then classified as
`MEMORY_SAMPLING_FAILURE` (or `MEMORY_PROCESS_DISAPPEARED` when the original
shipping process is proven exited) with an `INDETERMINATE` verdict. Cleanup
runs as independent stages (stop, bounded join, finalization, per-resource
close): one stage's failure never prevents the safe later stages, handles
close exactly once on normal and failure paths, and an escaping
`KeyboardInterrupt`/`SystemExit`/other `BaseException` always keeps its
exact identity. A mandatory process-sampler close or monitor-shutdown
failure prevents a `STABLE` verdict; a PDH GPU close failure is reported but
stays advisory. Failed process identity validation uses one checked rollback
close; any `CloseHandle` failure remains bounded mandatory secondary evidence.
GPU query construction publishes open state only after the counter binds;
every post-open `BaseException` attempts one checked `PdhCloseQuery` rollback,
and ordinary advisory GPU startup or rollback-close failures remain visible
as bounded report diagnostics while process monitoring continues without a
GPU retry.

The memory verdict is `NOT_REQUESTED`, `STABLE`, `UNSTABLE`, or
`INDETERMINATE`. `STABLE` requires at least five valid samples spanning at
least 60 seconds, preserved process identity, and every configured
working-set and private-commit threshold passed (equality passes; growth
stays signed). Percentage verdicts cross-multiply exact byte evidence against
the configured float's exact binary rational value: true equality passes and
any true above-limit value fails. A small tolerance is used only to verify
that stored display percentages match independently recomputed evidence; it
never affects the verdict comparison. The live acceptance boundary
additionally validates every
`STABLE` report against a central invariant checker — monitoring enabled
with full thresholds, preserved identity with a valid PID and executable,
both mandatory metrics present, passing, and fully evaluated, no mandatory
sampling/close/shutdown failure evidence, internally coherent
retained/omitted/total counts, and well-formed advisory GPU evidence — so a
malformed internally constructed `STABLE` report is rejected rather than
trusted. Memory evidence never interrupts an in-flight cycle: a process
exit, sampling failure, or irreversible peak breach only prevents admission
of the next cycle after the current cycle's cleanup and disappearance
verification finish, terminating the run as `MEMORY_PROCESS_DISAPPEARED`,
`MEMORY_SAMPLING_FAILURE`, or `MEMORY_STABILITY_FAILURE`. Cycle failures
always outrank memory and duration results; memory failures outrank
`SOAK_DURATION_REACHED`.

Disabled runs behave exactly as Mission 4G.5 and report an explicit
`NOT_REQUESTED` memory verdict.

## Mission 5: camera lane-perception foundation

Mission 5 adds a CARLA-independent, single-frame perception boundary:

```python
from perception import perceive_lanes

observation = perceive_lanes(telemetry_snapshot)
```

`perceive_lanes` consumes one coherent immutable, status-bearing
`TelemetrySnapshot` produced by `SensorSuite.snapshot()` or an equivalent
producer that attaches canonical sensor statuses to the same evidence cut.
A raw `TelemetryAggregator.raw_snapshot()` has no evaluated statuses and
therefore fails closed with `STATUS_MISSING`; perception does not infer or
invent freshness. Pixel processing starts only when that same snapshot
contains exactly one canonical `rgb_camera` status classified as continuous,
`ATTACHED`, `READY`, `FRESH`, and `HEALTHY`, plus one coherent immutable RGB8
frame. A continuous-camera status must have `event_count is None`; a populated
event count is incoherent event-sensor evidence and fails closed with
`STATUS_INCOHERENT`. Dimensions and stride must be positive built-in integers,
stride must equal `width * 3`, payload length must equal `width * height * 3`,
source frame and timestamps must be valid and finite, and the configured
image-pixel bound must not be exceeded. Missing, duplicate, stale, degraded,
failed, waiting, unavailable, destroyed, or malformed evidence fails closed
as an immutable `INPUT_UNUSABLE` observation with no lane geometry.

Image coordinates have their origin at the top-left. x increases rightward,
y increases downward, and all emitted point coordinates are normalized and
bounded to `[0.0, 1.0]`. Each boundary is fitted as
`x(y) = x_intercept + x_slope * y`; its two reported points are exactly its
intersections with the configured ROI top and bottom. The default ROI is the
normalized lower half `(left=0.0, top=0.50, right=1.0, bottom=1.0)`.

A complete lane reports a center offset at normalized y `0.90` as
`lane_center_x - 0.5`. Zero means the detected lane center coincides with the
image center, positive means it lies to the right, and negative means it lies
to the left. The dimensionless heading proxy is
`(center_x_at_roi_bottom - center_x_at_roi_top) / (roi_bottom - roi_top)`.
Positive means the image-space centerline moves rightward toward the bottom;
negative means it moves leftward. Neither value is world lateral error,
vehicle heading, steering angle, pose, localization, or world-metric lane
geometry.

Candidate extraction uses only the current RGB8 payload. Explicit white and
yellow channel thresholds are applied inside the ROI. Within each vertical
bin, candidate pixels are split into contiguous raster-row runs. Runs on
consecutive rows are matched one-to-one when their center displacement is no
greater than the one-row displacement implied by the configured maximum
slope. This preserves nearby parallel markings even when their aggregate x
projections overlap across a bin. Every resulting component retains a
deterministic bin-local cluster index, normalized x extent, row and pixel
support, and raster uncertainty. Its representative is an actual bright pixel
nearest its own component centroid, not an arithmetic midpoint between
clusters. Retention prefers row coverage and then pixel support with
coordinate tie-breaking. At most eight representatives survive per bin,
including for thick markings.

Boundary fitting is orientation-neutral: raw candidates are never partitioned
at image x `0.5`. Representatives form a bounded forward graph across at most
three bin steps, permitting no more than two consecutively missing bins. For
each bin gap, a representative retains at most four compatible outgoing links
and at most 12 links overall. Compatibility first enforces normalized
displacement from the configured slope and raster uncertainty. Once a partial
track contains two representatives, every extension must also agree with the
track's fitted prediction. This prevents a track from jumping between nearby
parallel structures even when both are inside the broader line-fit tolerance.

Dynamic programming retains at most four partial tracks per representative,
32 per vertical bin, and 64 completed tracks. A completed track is one
hypothesis seed, so no more than 64 of the 512-seed ceiling can be used by this
design. Each track contains at most one representative per vertical bin.
Fitting never replaces a track member with a nearby representative from a
different structure. Distinct-bin support, vertical span, RMS residual,
maximum individual error, and observed row coverage are enforced before a
boundary can reach pair selection. Near-equivalent fits are deterministically
deduplicated and at most 16 completed boundaries survive. Pair enumeration
directly stops at `MAX_PAIR_COMBINATIONS` (120 with the default fixed bounds),
rather than relying only on the hypothesis count. Work is bounded by the
configured image-pixel limit and these fixed graph, track, seed, hypothesis,
and pair caps. There is no random sampling, frame history, temporal smoothing,
mutable cache, learned model, image file, OpenCV, or SciPy.

The conservative default heuristic table is:

| Setting | Default |
|---------|---------|
| Normalized ROI | `(0.0, 0.50, 1.0, 1.0)` |
| White RGB minima / maximum channel spread | `(200, 200, 200)` / `55` |
| Yellow RGB minima / maxima / maximum R-G delta | `(160, 140, 0)` / `(255, 255, 140)` / `100` |
| Vertical bins / minimum supporting bins | `18` / `8` |
| Line inlier tolerance / maximum RMS residual | `0.035` / `0.025` normalized x |
| Maximum consecutive missing track bins | `2` |
| Track links per bin gap / total per representative | `4` / `12` |
| Partial tracks per representative / bin | `4` / `32` |
| Completed tracks / hypotheses / pair combinations | `64` / `16` / `120` |
| Minimum vertical span | `0.30` normalized y |
| Absolute boundary-slope range | `[0.05, 0.80]` |
| Plausible lane-width range | `[0.15, 0.80]` normalized x |
| Boundary confidence floor / detection threshold | `0.40` / `0.65` |
| Center-offset evaluation y | `0.90` |
| Maximum processed pixels | `1,000,000` |

Completed boundary pairs are first evaluated at the configured center row and
ordered by x: lower x is left and higher x is right. Neither boundary is
required to stay on one side of image x `0.5`. Only after ordering does the
detector require a negative left slope and positive right slope. Pair checks
at the ROI top, midpoint, center-evaluation y, and ROI bottom require left to
remain left of right and enforce inclusive lane-width bounds. Pair ranking
combines boundary confidence, distinct-bin support, vertical span, residual,
center proximity, width plausibility, and deterministic coefficient
tie-breaking; it is not based on residual alone.

Zero plausible boundaries yields `NOT_DETECTED`; one yields `PARTIAL`. Two
individually plausible but pair-invalid boundaries also yield `PARTIAL` with
no complete-lane geometry and aggregate confidence `0.0`, rather than exposing
their high individual scores as trustworthy pair confidence. A valid pair
below the inclusive detection threshold is `PARTIAL`; equality with the
threshold is `DETECTED`.

Ambiguity is explicit and bounded. A competing valid pair is materially
different when an evaluation-row boundary position or center offset differs
by at least `0.04`. If its deterministic quality is within `0.04`, aggregate
confidence within `0.08`, and minimum distinct-bin support within one bin of
the selected pair, the result is `PARTIAL` and its confidence is capped below
the detection threshold. Thus a second credible marking cannot turn several
structures into a high-confidence averaged boundary or increase clean-frame
confidence.

Boundary confidence combines bounded distinct-bin support, vertical coverage,
residual, slope plausibility, and observed cluster row coverage. Pair
confidence adds width and slope consistency, aggregate confidence never
exceeds either boundary, and ambiguity can only reduce it. These scores are
deterministic heuristic confidence, not calibrated probability or a measured
accuracy guarantee.

Observed support is a gate, not a small confidence bonus. The normalized
raster uncertainty is one half x pixel plus the configured maximum slope
projected across one half y pixel. Track prediction error is bounded by twice
that uncertainty (and no more than half the configured line-fit tolerance).
After fitting, every contributing bin's actual observed representative must be
within three times the raster uncertainty (and within the line-fit tolerance),
so a low RMS average cannot hide one excessive individual deviation. All
contributing bins are checked, their maximum missing run is two bins, and the
same prediction-consistent track supplies every check. Consequently pair
geometry, width consistency, and aggregate confidence never see a
continuity-invalid or materially unsupported boundary.

The perception call is stateless and deterministic: repeated calls on equal
snapshot/config values return equal observations. Source revision, camera
frame number, simulation timestamp, and host receive-monotonic timestamp are
copied from accepted evidence; no clock is read and no replacement time is
generated. Public configuration and observation records are frozen and
slotted, collections are tuples, and public numbers are ordinary Python
scalars. Returned observations retain no snapshot, camera frame, image bytes,
memoryview, NumPy object, sensor, callback, clock, exception, CARLA object, or
mutable collection. Temporary NumPy views and arrays exist only during the
call.

Mission 5 does not add a camera preview, construct or apply vehicle control,
implement lane keeping or autonomous driving, or target real-vehicle
deployment. It makes no certified ADAS, Level 3, real-world accuracy, or
measured CARLA accuracy claim. Live map, weather, lighting, occlusion, and
camera-placement performance remain unvalidated.

## Mission 5C: lane-perception hot-path performance

Mission 5C changes the implementation of the hot path without changing any
detection rule, threshold, comparison operator, bound, confidence formula,
enum, export, or public record from Mission 5.

The accepted implementation refitted a line from scratch for every candidate
track extension and again for every ranking key, so one ordinary 640x360 frame
performed thousands of small NumPy least-squares fits. Each partial track now
carries immutable scalar sufficient statistics — point count, the sums of y, x,
y squared, and x*y, the occupied-bin set, and the largest raster uncertainty of
its members. Appending one representative updates those accumulators in O(1)
using plain Python floats, without building a NumPy array, revisiting existing
members, or touching module-level state. Slope and intercept are derived from
the accumulators for the ordinary fast path.

Raw-moment and centred least-squares arithmetic are not bitwise
interchangeable at a decision boundary. For example, normalized 96x96 pixel
centres `(7, 20)` and `(11, 25)` produce scalar slope
`0x1.999999999999ap-1` but accepted centred slope
`0x1.999999999999bp-1`; at `max_abs_slope == 0.8`, those values make opposite
gate decisions. R3 therefore uses a conservative binary64 forward-error
envelope. It is derived from `gamma(k) = k*u / (1-k*u)`, `u = 2**-53`, the
64-point track bound, and quotient/intercept/prediction error propagation, with
an additional four-ULP comparison allowance. If fit definedness cannot be
proved, or the scalar slope or prediction error lies inside that envelope of
its configured boundary, the extension is decided with the accepted two-pass
centred fit and the original comparison operators. Clearly separated extension
decisions remain scalar and O(1).

The continuity check normally consults the carried coefficients rather than
refitting; only a numerically ambiguous gate uses the accepted fit. Its cheap
ranking key reuses coefficients and member identities already stored on the
track. The cheap scalar RMS residual is not treated as bitwise interchangeable
with the accepted Mission 5 residual. The accepted residual came from NumPy's
two-pass centred fit, and its last floating-point bits are an unthresholded
lexicographic ordering element.

At each hard cap, the implementation decorates every candidate once with the
cheap key. The first three key elements — support count, missing-bin count,
and vertical span — are fit-independent. If the equal-prefix group at the
cutoff extends to both sides of the cap, that bounded group is decorated with
authoritative keys to recover exact accepted membership. Every equal-prefix
group represented in the selected result is then authoritatively ordered, so
the returned tuple exactly matches the accepted Mission 5 order even for a
non-binding cap or a retained group away from the cutoff. Singleton groups need
no fit because the fit-independent prefix already fixes their position.

Authoritative keys are cached by immutable track identity in one dictionary
owned by one `_continuous_representative_tracks` call. A track is fitted for
cap ranking at most once during that call; later cap participation uses a
deterministic lookup. The cache is never iterated for ordering, is not
module-level, retains no frame or payload, and is discarded when the call
returns.

Numerically ambiguous extension gates use separate local dictionaries in each
scope where they are evaluated. The accepted-ranking and main extension caches
belong to one `_continuous_representative_tracks` call, while validation
re-walks use their own short-lived fallback caches. Accepted coefficients for
one immutable track are fitted at most once per cache scope and identity; the
same identity may be authoritatively fitted again by a later validation cache
during the same perception call. Every cache is discarded on return, retains
only track identities and coefficient pairs rather than a frame or image
payload, and provides no cross-frame retention.

Full correctness validation is unchanged and still runs once per retained
track. The authoritative closed-form NumPy fit supplies every published
coefficient, and distinct ordered bins, the skipped-bin bound, the complete
continuity re-walk, the RMS residual gate, the maximum individual-error gate,
endpoint projection, and observed row coverage are all still enforced before a
boundary can reach pair selection. Remaining fit calls are limited to
numerically ambiguous extension gates, final hypothesis validation,
cutoff-crossing groups needed for membership, and selected equal-prefix groups
needed for exact returned order. Ordinary clearly separated extension attempts
and cheap-key evaluation do not fit. Candidate extraction also narrows its
per-row run matching to the bounded window that the configured maximum slope
can reach, applying the identical acceptance test to an identical match list.

An isolated accepted Mission 5 package extracted from commit
`ede9826bb1842244010229c2dad018ab8877a14f` and the corrected implementation
were run in separate fresh processes with the same CPython 3.12 interpreter. A
deterministic 2,520-case R3 public differential covered 20 structured
regressions, 1,000 saturated/cluttered scenes, 500 unsaturated scenes, and
1,000 slope-boundary-targeted square scenes. It included 64, 80, 96, 112, 128,
160, 192, 200, 224, and 256 square inputs plus 160x90, 320x180, 640x360, and
800x600 rectangles; 2,080 cases were square. The generator cycled those sizes,
marking counts, signs, deterministic modular intercept/slope offsets, widths
1/3/5/11, default and 12-bin configurations, a `(0.05, 0.40, 0.95, 0.95)` ROI,
and one-ULP neighbours of the slope and line-fit thresholds. Exact public
serialization, using `float.hex()` for every float, produced zero differences
across 177,588 public leaf-field comparisons. This is evidence for the stated
corpus, not a proof over all possible inputs.

A separate 3,024,000-attempt extension differential used the same square and
rectangular sizes, 2/3/8/18-point tracks, positive and negative pixel-rational
slopes, configurable ULP-adjacent slope limits, degenerate probes, and
prediction errors concentrated below, at, above, and immediately adjacent to
tolerance. The pre-R3 scalar decisions disagreed with accepted arithmetic at
125,550 slope gates and 867,000 prediction gates; corrected complete extension
decisions had zero differences. Its 91.7% fallback-decision rate is deliberately
non-representative because almost every probe was generated on a numerical
boundary. Accepted fits were cached once for each of the 84,000 source tracks.

An internal ordered differential separately exercised 44,086 hard-cap events:
30,039 binding, 14,047 non-binding, and 23,582 with an equal-prefix group
crossing the cutoff. It included all three production cap sites, the compact
near-tie fixture, independently transformed near ties, forced 32- and 64-track
selections, named dense and adjacent-lane fixtures, 250 procedural saturated
scenes, and 100 unsaturated scenes. Selected sets, rejected sets, and complete
returned orders each had zero differences from
`sorted(tracks, key=accepted_key)[:limit]`. The per-call caches recorded 124,338
hits, 146,341 misses (including singleton keys that require no fit), and 139,505
authoritative cap-ranking fits.

The absolute counts below are measurements from the project's specific
deterministic fixture implementation. The scene labels and prose descriptions
do not uniquely define these workloads; exact reproduction requires the
corresponding fixture generator or fixture bytes. Fit calls on those fixed
640x360 fixtures were:

| Scene | Accepted fits | Boundary fallback fits | Exact-ranking fits | Final fits | Reduction |
|-------|--------------:|-----------------------:|-------------------:|-----------:|----------:|
| Clean two-marking lane | 1594 | 0 | 156 | 54 | 86.8% |
| Four-marking double line | 2918 | 0 | 314 | 64 | 87.0% |
| 30-marking adversarial | 3551 | 0 | 403 | 64 | 86.8% |
| 62-thin-marking adversarial | 2098 | 0 | 390 | 6 | 81.1% |

For that fixture implementation, zero extension-fallback fits occurred on the
ordinary fixtures, zero fits originated from the cheap `_track_quality_key`,
blank frames performed zero fits in both versions, and every dense scene above
reduced fits by at least 80%. On its targeted four-marking square blocker
fixture, fallback-decision rates were 0.92% at 96x96, 0% at 128x128, 1.46% at
160x160, 2.19% at 200x200, and 0.16% at 256x256; cached fallback-fit counts were
3, 0, 8, 22, and 1 respectively. Reproducing those absolute rates and counts
likewise requires the corresponding fixture generator or bytes.

Timing used `time.perf_counter_ns`, eight warm-ups and 30 measured calls per
scene in each of 16 alternating accepted/corrected pairs. Every block ran in a
fresh process with the same CPython 3.12 interpreter and disabled-GC policy;
all 32 blocks produced the identical fixture-byte digest
`5c70dbfb23e5dd9bc65e184e683573c08e30a0e724bee18264dc2cf87f1789bd`.
The table reports the median of the 16 round medians and the median of the 16
nearest-rank round p95 values:

| 640x360 scene | Accepted median | R3 median | R3 p95 | Relative result |
|---------------|----------------:|----------:|-------:|----------------:|
| Blank (independent reruns) | 1.76-1.79 ms | 1.85-1.90 ms | 2.52-2.59 ms | approximately 1.0x; 5-6% regression, within the <=10% limit |
| Clean two-marking lane | 56.04 ms | 24.79 ms | 27.12 ms | 2.26x |
| Four-marking double line | 102.15 ms | 39.85 ms | 43.15 ms | 2.56x |
| 30-marking adversarial | 130.84 ms | 58.99 ms | 62.65 ms | 2.22x |
| 62-thin-marking adversarial | 119.44 ms | 62.62 ms | 67.27 ms | 1.91x |

Single-block supplementary medians on the 96/128/160/200/256 square blocker
fixture were 25.54/27.30/31.60/32.55/31.74 ms for R3 and
64.99/73.06/83.78/85.95/86.74 ms for accepted Mission 5, or 2.54x through
2.73x faster. Those supplementary measurements use the same 8 warm-ups and 30
measured calls but are not the 16-pair primary benchmark.

These offline wall-clock measurements are from one machine and are not a
scheduling guarantee. The detector remains single-frame and stateless; no live
CARLA scheduling or accuracy measurement was made, and map, weather, lighting,
camera-placement, occlusion, and scheduling readiness remain unvalidated.

## Mission 6A: bounded perception runtime core

Mission 6A adds a CARLA-independent scheduling boundary around the accepted
Mission 5C single-frame detector:

```text
camera / telemetry
        |
        v
Mission 6A bounded runtime
        |
        v
Mission 5C lane perception
        |
        v
timestamped LaneObservation
```

The public runtime API is available without changing the existing top-level
single-frame perception exports:

```python
from perception.runtime import PerceptionRuntime, PerceptionRuntimeConfig

runtime = PerceptionRuntime(PerceptionRuntimeConfig(target_hz=10.0))
runtime.start()
runtime.submit(telemetry_snapshot)  # non-blocking scheduling path
result = runtime.latest_result
runtime.stop()
```

Construction reads no clock and creates no worker. `start()` explicitly creates
one non-daemon worker; perception never runs on the submitting producer. The
submission path validates canonical camera identity, samples the injected host
monotonic clock, and performs only a short locked state transition. Pixel
processing and every injected perception callable run outside the runtime lock.
Validated frame-order and timing evidence, including accepted `SampleStamp`
subclasses, is canonicalized into an exact `SampleStamp` containing built-in
immutable integer/float scalars before locked ordering comparisons. Validated
configuration subclasses are similarly reconstructed as exact immutable
runtime configuration. If the ingestion clock raises or returns a non-finite
value, that producer-side error propagates and shared state remains unchanged:
no received/accepted counter advances, no pending item is stored, and the
running runtime remains usable and stoppable.

### Latest-frame-wins bounded slot

The runtime has exactly one logical pending slot, represented by one optional
reference rather than a queue. If frame 101 is active and 102, 103, and 104 are
submitted, 104 is the only pending frame after the burst. Frames 102 and 103
increment `pending_frames_coalesced`; they are runtime scheduling replacements,
not source-sensor drops. There is no backlog draining and no pending or result
history. Submission work and retained scheduling state therefore do not grow
with a burst.

Camera `SampleStamp.frame` and `sim_time` are the ordering identity, matching
the telemetry camera aggregator. Snapshot revision is retained as evidence but
is never a competing frame number. An exact frame/time repeat is rejected as a
duplicate; an equal or lower frame with changed time, or any regressed
simulation time, is rejected as out of order. A higher frame at an equal
simulation time remains valid, matching the producer contract.

When exactly one canonical continuous `rgb_camera` status is present, its
monotonic `accepted_count` is optional source-sequence evidence. A strictly
increasing gap contributes the exact missing count to
`source_frames_inferred_dropped`. The first known count establishes a baseline
and creates no historical loss. Missing or regressed sequence evidence breaks
the inference chain; the runtime does not invent reset or wrap arithmetic.
CARLA world-frame-number jumps alone are not treated as camera drops. Runtime
coalescing is counted separately and never contributes to inferred source
loss. This inference requires coherent canonical producer evidence: the
telemetry aggregator advances the RGB camera `accepted_count` only when the
same camera ordering rule accepts and stores a frame. Synthetic snapshots that
advance `accepted_count` on a duplicate or rejected camera identity are outside
that contract and do not provide meaningful source-gap evidence.

### Timing and freshness

The immutable default configuration is:

| Setting | Default | Validated bound |
|---------|--------:|-----------------|
| Target processing budget | `10.0 Hz` | `[1/60, 1000] Hz` |
| Maximum pending age | `0.25 s` | `(0, 60] s` |
| Shutdown join timeout | `2.0 s` | `(0, 60] s` |

Bool, zero where positivity is required, negative, NaN, infinity, huge integers
that cannot convert to a finite float, wrong numeric types, and values outside
these finite bounds are rejected with `ValueError`. The corresponding
processing period is exactly `1 / target_hz`.

`target_hz` defines a deadline budget; it is not a pacing loop or rate limiter.
A fresh submission wakes the worker immediately and no runtime sleep is used.
After a completed invocation, only
`processing_latency_seconds > processing_period_seconds` is a deadline miss.
Equality passes. A miss increments metrics but never modifies or downgrades the
returned `LaneObservation`.

Two freshness layers remain distinct:

- Mission 5 source freshness is the immutable camera status evaluated in the
  submitted `TelemetrySnapshot`. The real `perceive_lanes` call keeps its
  existing attached/ready/fresh/healthy gate; Mission 6A neither replaces nor
  re-invents it. The runtime does not re-evaluate that frozen sensor status at
  processing time; scheduling delay is represented independently by pending
  age.
- Mission 6A pending age is host-monotonic time from runtime ingestion to the
  worker's processing start. Work is rejected before the perception callable
  when `pending_age_seconds` is greater than the configured maximum. Equality
  remains eligible.

All runtime timing is supplied by an injectable callable that defaults to
`time.monotonic`. Results record ingestion, processing start, processing
finish, pending age, and processing latency. Clock regression or non-finite
worker timing faults the runtime rather than manufacturing a duration.

### Result, metrics, lifecycle, and faults

`PerceptionRuntimeResult` is frozen and slotted. It preserves the exact
`LaneObservation`, canonical camera stamp, optional accepted-source sequence,
snapshot revision, and timing evidence. The runtime retains only the current
pending input, latest result, current fault, scalar metrics, and lifecycle
state. A result retains no telemetry snapshot, camera frame, RGB bytes, NumPy
object, or result history; the worker also releases its claimed snapshot before
waiting again.

Metrics are published as one immutable atomic view and distinguish:

- received, accepted, processed, successful, and failed perception inputs;
- duplicate and out-of-order rejection;
- stale pending rejection and claimed inputs abandoned by runtime
  infrastructure failure;
- inferred missing source frames;
- runtime-coalesced pending frames;
- pending work explicitly discarded by stop or fault;
- deadline misses plus last/maximum processing latency, last pending age, and
  the last processed source stamp.

`inputs_received` counts submissions that reach the runtime's locked receipt
transition. A submit attempt does not increment it when the runtime is not
`RUNNING`, when structural or evidence validation fails before receipt, or when
ingestion-clock sampling fails before receipt.

At quiescence each received input and each accepted input belongs to exactly
one terminal partition:

```text
inputs_received =
    inputs_accepted
  + duplicate_inputs_rejected
  + out_of_order_inputs_rejected

inputs_accepted =
    inputs_processed
  + stale_inputs_rejected
  + pending_frames_coalesced
  + pending_inputs_discarded
  + inputs_abandoned

inputs_processed = perception_successes + perception_failures
```

`inputs_abandoned` means an accepted input was claimed by the worker but could
not reach the ordinary processed/stale disposition because a processing clock,
result assembly, or unexpected worker-exit failure interrupted runtime
infrastructure. An unclaimed pending input removed when that fault is published
remains a `pending_inputs_discarded` item; it is never relabeled as abandoned.

`STOPPED`, `RUNNING`, `STOPPING`, and `FAULTED` are runtime lifecycle states,
not vehicle-supervisor states. Repeated start while running and repeated stop
while stopped are idempotent. Stop wakes an idle worker; if perception is
active, it permits that claimed call to finish while discarding the one
unclaimed pending item. Failure to join within the configured timeout leaves
`STOPPING` so a later stop can retry. Concurrent stop callers cannot finalize a
newly restarted epoch. A clean restart begins only after stop and resets the
pending slot, ordering baseline, result, fault, and all epoch metrics, so old
work cannot leak across epochs.

If the perception callable raises an ordinary `Exception`, no partial result is
published. The failed invocation is counted, bounded message and exception-type
evidence is retained without an exception or traceback, pending work is
discarded, and the runtime deterministically enters `FAULTED`. Clock and result
assembly faults are identified by their own fault stage rather than mislabeled
as callable failures and count their claimed item as abandoned. There is no
automatic restart loop. A faulted runtime is stopped explicitly before a new
epoch can start.

`BaseException` remains uncaught and may reach `threading.excepthook`. An outer
worker ownership finalizer nevertheless runs on every worker exit: an unexpected
exit publishes bounded generic `worker` fault evidence without retaining the
escaping object or traceback, abandons the claimed item exactly once, discards
any separate unclaimed pending item, and enters `FAULTED` before the thread is
dead. Consequently `submit()` cannot continue accepting work for a dead worker.
`stop()` still joins and releases that worker deterministically; restart retains
the existing explicit stop-before-start rule and begins a fresh epoch.

### Scope and validation boundary

Importing `perception.runtime` or `perception`, and constructing a runtime, do
not read a clock, create a thread, mutate the filesystem or environment, or
eagerly import NumPy, CARLA, pygame, OpenCV, or SciPy. NumPy can load later only
through legitimate Mission 5 pixel processing. Explicit start/stop with an
injected fake perception callable remains fully offline.

Mission 6A implements no temporal lane tracking, smoothing, safety supervisor,
scenario runner, fault-injection framework, exhibition GUI, steering,
throttle, braking, autopilot, actor spawning, world loading, or other vehicle
control. Its validation is offline only; no live CARLA scheduling or accuracy
claim is made. Live CARLA integration is deferred to Mission 6B.

## Mission 6B: bounded live perception integration

Mission 6B connects the existing RGB camera boundary to the Mission 6A
runtime without adding another camera, callback queue, worker, or telemetry
model:

```text
CARLA RGB Image callback
        |
        | copy BGRA source and build immutable RGB8 CameraFrame
        v
TelemetryAggregator.submit_camera_frame_and_snapshot()
        |
        | exact accepted CameraFrame + exact simulation timestamp
        v
PerceptionLiveBridge (admission and in-flight accounting only)
        |
        | non-waiting runtime.submit(snapshot)
        v
PerceptionRuntime (one worker, pending capacity exactly one)
```

### Exact camera association and callback boundary

Camera acceptance, source ordering, latest-frame publication, health counters,
revision advancement, and raw snapshot construction are one aggregator-locked
operation. The accepted snapshot retains the exact input `CameraFrame` and its
exact `SampleStamp`; snapshot revision remains coherent evidence and is never
used as camera identity. A later concurrent or reentrant camera publication
therefore cannot replace the camera associated with an earlier callback.
Rejected, duplicate, out-of-order, and post-shutdown frames produce no
perception submission.

`SensorSuite` decorates that atomic raw snapshot with the existing immutable
freshness statuses while preserving the exact camera reference and source
stamp. Clock sampling and the optional camera consumer both run outside suite,
sensor, aggregator, and bridge locks. With no consumer configured, the prior
camera publication path and all Mission 4G/5 behavior remain unchanged.

Mission 6B-R1 used reservation generation ordering to handle the overlap where
an older call completed after a newer call, but that proved only one orientation
and could misclassify the mirror sample/commit ordering as clock rollback.
Mission 6B-R2 retains only a bounded scalar observation generation, effective
time, and non-overlap cutover. A rollback comparison now occurs only when the
observation was reserved after the reference state had committed. Every
already-reserved observation is conservatively overlapping, cannot manufacture
a rollback or degrade sensor health, and may only advance effective time. A
genuinely later sequential regression remains detected and clamped. Clock
invocation and canonicalization remain outside `_snapshot_lock`.

A camera snapshot consumer is valid only when the RGB camera is enabled. The
disabled-RGB plus non-`None` consumer combination raises `ValueError` during
`SensorSuite` construction, before any sensor adapter, actor, listener, callback
registration, consumer lookup, or consumer retention can occur. Disabled RGB
without a consumer and enabled RGB with a consumer retain their prior behavior.
Construction also copies the five enabled flags and freshness policies into
built-in booleans and exact base policy records. The snapshot critical section
uses only that trusted immutable evidence; no caller-controlled configuration
attribute or freshness-policy hook executes under `_snapshot_lock`.

The live camera callback is limited to copying the CARLA buffer, BGRA-to-RGB8
conversion, immutable record construction, atomic publication, freshness
decoration, and bridge submission. It performs no lane processing, waiting,
sleeping, actor RPC, GUI work, result interpretation, vehicle control, or
network operation. It retains no CARLA `Image`, `raw_data` view, NumPy array,
mutable source buffer, actor, world, or client. Ordinary consumer failures are
isolated after successful camera parsing; `BaseException` is not swallowed.

### Bounded bridge and ownership

`perception.live_bridge.PerceptionLiveBridge` is CARLA-free and NumPy-free. It
owns no worker, queue, pending slot, snapshot history, result history, or frame
archive. Its one-shot lifecycle is `OPEN`, `CLOSING`, `CLOSED`, or `FAULTED`.
Each open submission registers one in-flight call, releases the bridge lock,
calls `PerceptionRuntime.submit`, samples completion latency on the same host
monotonic basis as camera receipt, then finalizes immutable scalar metrics and
notifies drain waiters. Runtime rejection is a normal bounded result. Once the
runtime returns a valid boolean, that acceptance or rejection is the admitted
call's single terminal accounting class. A later completion-clock or
instrumentation fault remains truthful bridge fault evidence but cannot count
the same admission again as a submission failure.

An ordinary submission or bridge-infrastructure exception retains at most one
bounded immutable fault record, never an exception or traceback, and closes
admission. `BaseException` propagates only after in-flight accounting and
waiter notification are complete. Non-finite or regressed completion timing
faults deterministically. `close_admission()` gates first; `drain(timeout)`
waits only for already-admitted calls. A normal timeout preserves `CLOSING`,
allowing a later bounded retry to finish in `CLOSED`. `FAULTED` is sticky public
lifecycle truth: admission remains closed and drain, close, timeout, retry, and
in-flight completion remain usable, but successful teardown never rewrites the
historical fault as `CLOSING` or `CLOSED`.

The outer live harness owns the runtime, bridge, suite, and ego. `SensorSuite`
does not start, stop, or own the runtime or bridge, and releases its optional
consumer reference during destruction. Every cycle constructs fresh instances
and uses this cleanup order even when an earlier stage fails or times out:

```text
1. close bridge admission
2. boundedly drain admitted bridge calls
3. destroy SensorSuite (gate, tick unregister, sensor stop/destroy)
4. stop PerceptionRuntime
5. destroy the owned ego
6. verify all owned actor IDs disappear
7. release retained live references
8. verify no runtime worker remains
```

### Separate live observation entry and validation boundary

`tests/integration_perception_runtime.py` is an inert, separately invoked live
observation entry. Importing or normally collecting it does not import CARLA,
NumPy, the runtime, bridge, or sensor suite; connect to port 2000; spawn an
actor; start a worker; or mutate the environment. Only the exact
`CARLA_INTEGRATION=1` opt-in loads live dependencies and connects to an already
running compatible CARLA server. The harness never launches CARLA. Its bounded
configuration supports a frame target or duration, cycle count, runtime target
frequency, pending-age limit, shutdown timeout, and first/final-only report
retention.

The live cycle success predicate also requires at least one processed result
whose input passed the accepted Mission 5 RGB attached/ready/fresh/healthy
status gate. Runtime invocation success alone is insufficient, so a cycle
cannot report success while required RGB evidence is degraded by a host-clock
rollback or is otherwise unusable.

### Mission 6B final acceptance

The final acceptance environment used the canonical Python 3.12.10
interpreter, CARLA client 0.9.16 and server 0.9.16, and Town10HD_Opt.
Independent offline acceptance passed Claude R2 rereview, the full offline
suite at 1116/1116, the focused Mission 6B collection at 86/86, and the legacy
collection at 1030/1030.

Controlled live acceptance also passed. Stage A completed 1/1 fresh cycle with
10 admitted, 10 processed, and 10 perception successes. Stage B completed 3/3
additional fresh cycles, for four total fresh lifecycle cycles. Stage C
completed a 60.0-second bounded stability run with 599 admitted, 599 processed,
and 599 perception successes; rejections, coalescing, perception failures, and
deadline misses were all zero. Across the controlled live acceptance,
bridge/runtime component faults, cleanup failures, remaining acceptance-owned
actors, and surviving runtime workers were all zero.

Mission 6B remains the accepted live-integration baseline. Mission 7 is the
accepted offline temporal-tracking mission described below; it has not changed
the accepted Mission 6B live result.

## Mission 7: bounded temporal lane tracking

Mission 7 adds a deterministic temporal interpretation of the accepted
Mission 5 image-space lane observation without adding vehicle control:

```text
TelemetrySnapshot
        |
        v
Mission 5 perceive_lanes (one independent LaneObservation)
        |
        v
Mission 7 TemporalLaneTracker (one bounded temporal estimate)
        |
        v
future Mission 8 safety supervision
```

`perception.temporal.TemporalLanePipeline` is the opt-in callable adapter for
this path. It invokes the configured Mission 5 callable, supplies the canonical
camera `SampleStamp` from the same snapshot to the tracker, and returns a
`TemporalLaneObservation`. The adapter does not command the vehicle.

### Temporal states and dropout policy

The public temporal estimate is frozen, slotted, payload-free, and has five
states:

- `UNINITIALIZED`: no accepted usable temporal evidence exists, including
  immediately after construction or reset.
- `TRACKING`: the current detected or partial observation contains usable lane
  evidence. Only geometry actually present in that observation is updated.
- `COASTING`: the current raw state is the ordinary Mission 5
  `NOT_DETECTED` result, and a prior track remains within all configured miss,
  source-simulation-age, and minimum-confidence limits. The estimate is marked
  extrapolated and its confidence is strictly decayed.
- `LOST`: no trustworthy track remains. This includes an initial ordinary miss
  and expiration of any miss, age, or confidence bound. Geometry is hidden and
  usable confidence is zero.
- `INPUT_UNUSABLE`: the current sensor/input evidence is unhealthy or
  structurally unusable. This state is exposed immediately, is never relabeled
  as tracking or coasting, hides guidance geometry, and reports zero usable
  confidence.

An ordinary detection dropout and unusable input are intentionally different.
Only `NOT_DETECTED` may coast, and then only while every bound remains valid;
`INPUT_UNUSABLE` fails safe immediately. A newer usable detection or partial
observation recovers deterministically to `TRACKING`.

The immutable default policy is:

| Setting | Default | Validated bound |
|---------|--------:|-----------------|
| EMA gain | `0.35` | `(0, 1]` |
| Consecutive ordinary misses | `2` | built-in integer `[0, 1,000,000]` |
| Maximum coast age | `0.30 s` | `(0, 86,400] s` |
| Per-miss confidence decay | `0.75` | `(0, 1)` |
| Minimum coast confidence | `0.10` | `(0, 1]` |

### Smoothing and confidence

The tracker uses first-order exponential smoothing, not a prediction filter.
For each present left or right boundary it applies the configured EMA to the
Mission 5 line coefficients `x_intercept` and `x_slope`. A complete detected
lane also smooths `center_offset` and `heading_proxy`. Partial observations
update only boundaries actually present and never manufacture a missing side
or complete center geometry. One tracker lifecycle assumes the upstream Mission
5 ROI and geometry convention remain fixed because observations do not carry a
configuration identity; complete geometry reinitializes as a unit after partial
evidence so one-sided history is not mixed into a new complete lane.

Confidence remains conservative. A detected update may recover confidence but
never above the current raw evidence; a partial update cannot increase the
previous confidence. Every coast multiplies track and boundary confidence by
the strict decay factor, so missing evidence cannot make the estimate more
optimistic. `LOST` and `INPUT_UNUSABLE` expose no guidance geometry and zero
usable confidence.

### Source ordering and fail-safe unstamped input

Stamped updates use the exact Mission 6 camera-order contract. The tracker
canonicalizes built-in scalar evidence before retaining it. Exact frame and
simulation-time repeats are duplicate rejections with no state mutation. An
equal or lower frame with changed time, or any regressed simulation time, is
out of order; a higher frame at equal simulation time remains valid. Snapshot
revision and host receive time are evidence, not competing source identity.

Mission 5 deliberately omits source fields when it emits `INPUT_UNUSABLE`.
During normal pipeline use the adapter recovers the canonical camera stamp from
the same snapshot. A directly supplied unstamped `INPUT_UNUSABLE` observation
still fails safe immediately but does not advance the source-order baseline.
Malformed, non-finite, or mismatched explicit source evidence raises instead
of silently falling back. All rejection and transition accounting remains in
bounded scalar metrics.

### Bounded state, reset, and runtime epochs

`TemporalLaneTracker` retains one canonical geometry estimate, one immutable
latest result, source/order scalars, miss/age/confidence scalars, and bounded
counters. Its production state is O(1): there is no frame or observation
history, queue, deque, replay buffer, clock, lock, worker, or additional
thread. Source simulation time supplies temporal age. The tracker is explicitly
single-consumer; `update()` and `reset()` must not be called concurrently.
The pipeline adds only one latest accepted payload-free carrier so a direct
duplicate or out-of-order call can return a coherent prior raw/temporal pair;
it never retains the input snapshot, RGB bytes, or a result history.

`TemporalLaneTracker.metrics` is bounded diagnostic evidence, not a
concurrently atomic multi-counter snapshot. Each returned record is immutable,
but its several scalar counters are read without synchronization, so an
unsupported concurrent reader can observe a transient accounting mismatch while
an update is in progress. Consume metrics from the owning consumer context or
after quiescence when the documented partition identities must hold exactly.
The immutable temporal estimate itself is unaffected.

`reset()` clears geometry, source ordering, last-usable time, misses, and the
latest estimate back to `UNINITIALIZED`; lifetime scalar counters, including
the reset count, remain truthful. Repeated reset is safe.

**Reset or recreate the temporal pipeline for every new runtime epoch.** This
is a required lifecycle contract, not a suggestion. Restarting
`PerceptionRuntime` resets the runtime's own ordering baseline, metrics, result,
and fault evidence, but it cannot reset state hidden inside an arbitrary
callable. A pipeline carried into a new source epoch without reset keeps the
previous epoch's source-order baseline; if the new epoch's frames restart at or
below that baseline, the tracker rejects them as out of order and the pipeline
replays its cached prior-epoch carrier until the new frame numbers overtake the
retained baseline. Within one correctly constructed Mission 6 runtime epoch this
is unreachable, because the runtime's own duplicate/out-of-order gate is
identical to the tracker's and rejects such input before perception runs. Create
one pipeline per runtime epoch, or call `pipeline.reset()` while that runtime is
stopped.

There is one deliberate publication boundary to preserve, and it reinforces the
rule above. The pipeline updates its tracker before the runtime samples the
processing-finish clock and assembles `PerceptionRuntimeResult`. If that finish
clock or result construction fails, the runtime faults and publishes no new
runtime result even though the pipeline estimate has already advanced, so
`latest_result` can lag the tracker by one observation. The runtime becomes
`FAULTED` and refuses to start until it is stopped, so normal continuation is
prohibited. **Always reset or recreate the pipeline after a faulted runtime
epoch** before reusing it. Cross-thread consumers should use the immutable
temporal estimate embedded in a successfully published runtime result rather
than racing the single-consumer tracker directly.

### Atomic Mission 6 integration

`TemporalLaneObservation` is a frozen, slotted `LaneObservation` subtype. Its
inherited fields remain exactly the raw Mission 5 observation; smoothed evidence
is carried separately in its required `temporal_estimate` field. The accepted
runtime already admits `LaneObservation` subclasses, so a successful runtime
publication associates the raw and temporal views atomically without changing
`PerceptionRuntimeResult`.

This integration is opt-in through `perception_callable=TemporalLanePipeline()`.
The default Mission 6 runtime still calls raw `perceive_lanes`; its pending
capacity remains exactly one, latest-frame-wins coalescing and all accounting
partitions remain unchanged, and the Mission 6B bridge keeps its existing
boolean admission/drain contract. Runtime-coalesced or runtime-stale inputs do
not reach the pipeline and therefore are not fabricated as temporal misses or
updates. Mission 7 adds no worker or bridge lifecycle stage.

### Mission 7 offline acceptance

Mission 7 adds 70 deterministic tests across two new discoverable files and is
accepted for commit after an independent adversarial review returned **PASS**
with 0 blocker and 0 major findings, and 3 accepted minor findings recorded
below.

| Evidence | Result |
|----------|--------|
| Mission 7 suites | 70/70 pass |
| Critical M5/M6/sensor collection | 466/466 pass |
| Full offline discovery | 1,186/1,186 pass |
| Static test methods, all files | 1,188 |
| Statically discoverable and executed | 1,186 |
| Structurally excluded live methods | 2 |
| Independent boundedness stress | 1,000,000 updates, O(1) retained state |
| CARLA | not contacted for Mission 7 |

Failures, errors, and skips were all zero. The two excluded methods live in
`tests/integration_perception_runtime.py` and `tests/integration_sensor_suite.py`,
which the `test_*.py` discovery pattern does not match; they remain the
opt-in live harnesses. The counts reconcile exactly with the Mission 6B
baseline of 1,118 static and 1,116 discoverable: all 70 additional methods are
Mission 7 tests. The independent boundedness stress also covered long
`NOT_DETECTED` and `INPUT_UNUSABLE` streams, a duplicate/out-of-order reject
flood, and repeated resets, retaining no history, container, or exception.

### Accepted Mission 7 limitations

Three minor findings are accepted as documented backlog rather than remediated
in Mission 7. None is reachable inside one correctly constructed Mission 6
runtime epoch.

- **Runtime-epoch pipeline reuse.** Reusing a `TemporalLanePipeline` or
  `TemporalLaneTracker` in a new runtime epoch without reset retains the old
  source-order baseline and can republish a cached prior-epoch carrier after a
  tracker rejection. The accepted contract is to reset or recreate the temporal
  pipeline for every new runtime epoch, as described above.
- **Advance before publication fault.** The tracker can advance and the runtime
  can then fault during finish-clock sampling or result construction before
  publishing the corresponding result, so `latest_result` may lag the tracker by
  one observation. The runtime becomes `FAULTED`; a reset or fresh pipeline for
  the next epoch restores coherent state. This reinforces the reuse rule above.
- **Metrics are not concurrent-atomic.** `TemporalLaneTracker.metrics` reads
  several scalar counters without synchronization, so an unsupported concurrent
  reader may observe a temporary accounting mismatch. The tracker remains
  explicitly single-consumer and the immutable temporal estimate is unaffected.

Two Mission 7 boundaries remain the consumer's responsibility. The tracker does
not age autonomously: coast age advances only when a newer observation arrives,
so holding a result indefinitely never expires it on its own. Every estimate
therefore carries its `source_stamp`, `snapshot_revision`, `track_age_seconds`,
and `extrapolated` flag, and later supervision must enforce result freshness
from that evidence.

This is offline acceptance only. No controlled live Mission 7 acceptance has
been performed, and the accepted Mission 6B live evidence above must not be read
as temporal-tracking acceptance or as production-road readiness. Mission 7
contains no steering, throttle, braking, autopilot, scenario, dashboard, safety
supervisor, or other vehicle-control behavior. Missions 8 and 9 are described
below as independently reviewed and accepted offline implementations; Missions
10-19 remain unimplemented.

## Mission 8: bounded safety supervision

Mission 8 answers exactly one question: **may autonomous control be permitted
right now?** It is a permission layer, not a controller. It issues no steering,
throttle, brake, or `apply_control` call, owns no CARLA actor, and adds no
worker, queue, timer, or lock.

```text
CARLA / sensors
        |
        v
Mission 5 perceive_lanes
        |
        v
Mission 6 bounded PerceptionRuntime (+ Mission 6B bridge)
        |
        v
Mission 7 TemporalLanePipeline (atomic temporal carrier)
        |
        v
Mission 8 SafetySupervisor  --->  immutable SafetyDecision
        |
        v
future Mission 9/10 controllers (must obey autonomy_allowed)
```

The supervisor lives in the `safety` package and consumes only published,
immutable evidence:

- one `PerceptionRuntimeResult` whose observation is a Mission 7
  `TemporalLaneObservation`,
- one bounded `ComponentHealth` record copied out of the Mission 6A runtime and
  Mission 6B bridge; the argument is optional per call, but an epoch that has
  never received one holds component health *unknown* and refuses autonomy,
- one caller-supplied host-monotonic safety time, sampled after the result is
  read.

It publishes one frozen, slotted `SafetyDecision`. Downstream controllers must
require `decision.autonomy_allowed is True` before issuing any vehicle command.
Mission 8 deliberately provides no such command.

### Safety states and reasons

| State | Meaning | `autonomy_allowed` |
|-------|---------|--------------------|
| `INITIALIZING` | no validated fresh usable result yet | `False` |
| `NOMINAL` | fresh, coherent, strong tracking evidence | `True` |
| `DEGRADED` | fresh, coherent, reduced-quality evidence inside permitted bounds | `True` |
| `RECOVERING` | evidence is healthy again after a soft loss, streak incomplete | `False` |
| `FAIL_SAFE` | safety cannot be established | `False` |

Every reason belongs to exactly one state, and `SafetyDecision` validates the
state/reason/permission triple in `__post_init__`. No code path can publish a
permitted decision under an unsafe explanation, and only a `FAIL_SAFE` decision
may report `latched=True`.

| Reason | State | Latches |
|--------|-------|---------|
| `STARTUP_NO_RESULT` | `INITIALIZING` | no |
| `NOMINAL_TRACKING` | `NOMINAL` | no |
| `DEGRADED_PARTIAL_TRACKING` | `DEGRADED` | no |
| `DEGRADED_COASTING` | `DEGRADED` | no |
| `DEGRADED_LOW_CONFIDENCE` | `DEGRADED` | no |
| `RECOVERY_PENDING` | `RECOVERING` | no |
| `CONFIDENCE_BELOW_THRESHOLD` | `FAIL_SAFE` | no |
| `RESULT_STALE` | `FAIL_SAFE` | no |
| `TEMPORAL_LOST` | `FAIL_SAFE` | no |
| `TEMPORAL_INPUT_UNUSABLE` | `FAIL_SAFE` | no |
| `TEMPORAL_UNINITIALIZED` | `FAIL_SAFE` | no |
| `COMPONENT_NOT_READY` | `FAIL_SAFE` | no |
| `RUNTIME_FAULT` | `FAIL_SAFE` | **yes** |
| `BRIDGE_FAULT` | `FAIL_SAFE` | **yes** |
| `IDENTITY_MISMATCH` | `FAIL_SAFE` | **yes** |
| `EVIDENCE_MALFORMED` | `FAIL_SAFE` | **yes** |
| `CLOCK_REGRESSED` | `FAIL_SAFE` | **yes** |

Each decision also carries the accepted source stamp and snapshot revision, the
evaluation monotonic time, the result age, the Mission 7 temporal state and
confidence, the recovery streak, and the latch flag. No result history is kept.

### Configuration

One immutable validated record. The defaults are derived from the accepted
Mission 5 and Mission 6 contracts rather than copied from Mission 7's
miss/coast policy, because Mission 7 decides whether *its track* is
`TRACKING`/`COASTING`/`LOST` while Mission 8 independently decides whether the
resulting evidence is safe enough and fresh enough to permit autonomy.

| Setting | Default | Validated bound | Why this value |
|---------|--------:|-----------------|----------------|
| `max_result_age_seconds` | `0.35 s` | `(0, 60] s` | one accepted 10 Hz runtime period (`0.10 s`) plus the entire accepted Mission 6A pending budget (`0.25 s`) |
| `min_nominal_confidence` | `0.65` | `[0, 1]` | the accepted Mission 5 `detection_confidence` gate |
| `min_degraded_confidence` | `0.40` | `[0, 1]` | the accepted Mission 5 `min_boundary_confidence` gate |
| `required_healthy_results` | `3` | `[1, 1,000,000]` | about `0.30 s` of uninterrupted 10 Hz strong tracking |

`min_degraded_confidence` may not exceed `min_nominal_confidence`. Bool, NaN,
`+inf`, `-inf`, wrong numeric types, non-`int` sample counts, and out-of-range
values are rejected with `ValueError`. Caller-owned configuration records and
hostile `numbers.Real` values are canonicalized to exact built-ins once, before
retention, so a value that changes between reads cannot influence a later
decision.

### Result freshness: the contract Mission 7 cannot enforce

Mission 7 does not age autonomously. Its coast age advances only when a newer
observation arrives, so holding a good result forever never expires it. Mission
8 closes that gap:

```python
decision = supervisor.update(result, now=time.monotonic(), health=health)
...
# perception published nothing new; safety time still has to move
decision = supervisor.evaluate(now=time.monotonic(), health=health)
```

`evaluate` advances safety time with **no perception input at all**. A
previously `NOMINAL` or `DEGRADED` decision therefore becomes `FAIL_SAFE` once
its age exceeds the configured maximum, without requiring another Mission 7
observation, and no number of further ticks can restore it.

Age is measured from the accepted result's camera receive time
(`source_stamp.monotonic`), which is the true age of the world evidence a
controller would act on. **Callers must supply `now` from the same host
monotonic base** — `time.monotonic`, as used by the Mission 6A runtime and
Mission 6B bridge by default. A safety time earlier than the evidence it is
being compared against is a wiring error, not a stale reading, and hard-latches
`CLOCK_REGRESSED` rather than producing a negative age.

#### Required caller order

The runtime publishes concurrently, so the order of these two steps matters:

```python
published = runtime.latest_result      # 1. read the published result FIRST
now = time.monotonic()                 # 2. THEN sample safety time
decision = supervisor.update(published, now=now, health=health)
```

Sampling `now` *before* reading `latest_result` leaves a window in which a frame
published in between carries `source_stamp.monotonic > now`. That is a negative
age, and it hard-latches `CLOCK_REGRESSED` until `reset()`. Reading the result
first guarantees the invariant the supervisor requires:

```text
now >= result.source_stamp.monotonic
```

Any wiring that guarantees that inequality is acceptable; the read-then-sample
order is simply the cheapest way to get it. `now` must also be finite and
non-decreasing across calls on one supervisor epoch.

The boundary is inclusive: age exactly equal to `max_result_age_seconds` is
still fresh, matching the accepted Mission 6A pending-age and Mission 7
coast-age conventions. One ULP beyond it is stale; a regressed safety clock
hard-latches.

### Temporal state policy

| Mission 7 state | Raw state | Mission 8 outcome |
|-----------------|-----------|-------------------|
| `TRACKING` | `DETECTED` | `NOMINAL` at or above the nominal threshold; `DEGRADED` at or above the degraded threshold; otherwise `FAIL_SAFE` |
| `TRACKING` | `PARTIAL` | `DEGRADED` at or above the degraded threshold; otherwise `FAIL_SAFE` |
| `COASTING` | `NOT_DETECTED` | `DEGRADED` at or above the degraded threshold; otherwise `FAIL_SAFE` |
| `LOST` | any | `FAIL_SAFE` |
| `INPUT_UNUSABLE` | any | `FAIL_SAFE` |
| `UNINITIALIZED` | any | `FAIL_SAFE` |

Coasting is permitted only while Mission 7 itself still reports `COASTING`, the
result is fresh under Mission 8's own bound, confidence still meets the degraded
threshold, and no component or integrity fault exists. `LOST` and
`INPUT_UNUSABLE` are never reinterpreted as usable because earlier history was
good. Confidence comparisons are inclusive at the threshold.

### Result identity consistency gate

Mission 7 already pairs raw and temporal evidence atomically inside
`TemporalLaneObservation`. Mission 8 adds the comparison Mission 7 cannot make:
between the **outer** `PerceptionRuntimeResult` identity and the carrier it
transports. Before any ordering or policy decision, the supervisor requires

- the carrier's snapshot revision to equal the result's snapshot revision,
- the carrier's source frame, simulation timestamp, and receive time to equal
  the result's `source_stamp` (a raw carrier may omit them only when its state
  is `INPUT_UNUSABLE`, exactly as Mission 5 emits it),
- the temporal estimate's snapshot revision and source stamp to equal the same
  result identity,
- the carrier and the estimate to agree on the raw detection state.

Any disagreement is an invariant violation: it fails closed immediately, latches
`IDENTITY_MISMATCH`, and never becomes `NOMINAL` or `DEGRADED`. A result that is
not a Mission 7 carrier at all, or whose fields are structurally malformed,
latches `EVIDENCE_MALFORMED`.

**This gate runs before the ordering gate on purpose.** A prior-epoch carrier
republished under a restarted epoch's identity usually also looks out of order,
so checking order first would silently mask the invariant violation as a routine
rejection.

#### Mission 7 F1 protection: cross-epoch cached carrier

The accepted Mission 7 minor finding is that a `TemporalLanePipeline` reused
across runtime epochs without reset keeps its old source-order baseline, rejects
the new epoch's restarted frames as out of order, and republishes its cached
prior-epoch carrier. Mission 6 then wraps that stale carrier in a *new* runtime
result carrying the *new* frame's identity. Mission 8 detects exactly that
mismatch and latches, and
`tests/test_integration_safety_supervisor.py` reproduces the scenario end to end
through the real runtime and pipeline rather than by hand-building the pair.

#### Mission 7 F2 protection: advance before publication

Mission 7 can advance its tracker and Mission 6 can then fault before publishing
the corresponding result, so `latest_result` may lag the tracker by one
observation. Mission 8 does not reach into tracker internals to compensate. It
consumes only published immutable results plus component fault evidence, so a
runtime fault forces an immediate latched `FAIL_SAFE`, and even with no fault
evidence at all the freshness bound expires the last published result on its own.

### Component faults and the hard latch

`ComponentHealth` is a bounded immutable copy of the Mission 6 runtime and
Mission 6B bridge health; the supervisor never retains the components
themselves. `ComponentHealth.from_components(runtime, bridge)` copies the public
state and fault presence once per control step.

- A faulted runtime state **or** a present runtime fault latches `RUNTIME_FAULT`.
- A faulted bridge state **or** a present bridge fault latches `BRIDGE_FAULT`.
- A runtime that is not `RUNNING`, or a bridge that is no longer `OPEN`, is a
  soft `COMPONENT_NOT_READY` loss rather than a latch.

State and fault presence are two independent lock-protected reads, so a torn
read can only *add* fault evidence — it fails closed. Bridge fault evidence
supplied without a bridge state is contradictory and raises.

#### `health` omitted means "no new sample", never "healthy"

Component health is evidence the caller supplies. The `health` argument is
optional **per call**, but omitting it never asserts that the components are
fine:

| Situation | Component readiness | Autonomy |
|-----------|---------------------|----------|
| epoch has never received a `ComponentHealth` | **unknown** | **not permitted** |
| healthy record supplied | ready | permitted if the evidence also qualifies |
| healthy supplied, then omitted | last known ready persists | still permitted |
| not-ready supplied, then omitted | last known not-ready persists | not permitted |
| fault supplied, then omitted | latched | not permitted |
| after `reset()` | back to **unknown** | **not permitted** |

Until a supervisor epoch has actually observed component health, a perfectly
fresh, coherent, strong result yields `COMPONENT_NOT_READY` with
`autonomy_allowed = False`. Unknown is not healthy: a caller that forgets to
wire health is never handed autonomy by default, and a runtime fault followed by
a restart inside the same CARLA session — where frame numbers keep rising —
cannot silently re-permit autonomy either.

Unknown health takes the existing soft `COMPONENT_NOT_READY` path, so it arms
the normal recovery hysteresis exactly like any other not-ready component: if a
health-less call happens first, the first genuinely healthy sample resumes
through the usual `required_healthy_results` streak rather than jumping straight
to `NOMINAL`. Ticking before any result at all still reports `INITIALIZING`,
because the startup explanation is checked first.

A correctly wired startup that supplies health immediately has no recovery
delay: its first otherwise-qualifying result leaves `INITIALIZING` directly.

Hard failures stay latched until an explicit `reset()`. A later perfectly
healthy result cannot clear a runtime fault, a bridge fault, an identity
mismatch, malformed evidence, or a regressed clock.

### Soft-loss recovery hysteresis

A soft loss — transient `LOST`, `INPUT_UNUSABLE`, low confidence, a stale
result, or a component that is not ready, including one whose health has never
been observed this epoch — arms recovery and clears the healthy streak.
While armed, the supervisor reports `RECOVERING` with
`autonomy_allowed = False` until it has seen `required_healthy_results`
consecutive strictly newer results that are fresh, coherent, fault-free, and
strong enough to be `NOMINAL` on their own.

Only such a result advances the streak. A duplicate, an out-of-order result, a
plain `evaluate` tick, a partial or coasting result, and a detected result below
the nominal confidence threshold all fail to advance it; any accepted result
that is not a strong healthy sample resets the streak to zero, because the
requirement is explicitly consecutive. Autonomy resumes on the exact Nth strong
sample. Startup is not a soft loss: the first validated fresh usable result
leaves `INITIALIZING` directly.

### Source order and duplicates

Mission 8 reuses the accepted Mission 6 source-order semantics against one O(1)
retained source identity. An exact frame and simulation-time repeat is an
idempotent duplicate rejection; a frame at or below the baseline, or any
regressed simulation time, is an out-of-order rejection; a higher frame at equal
simulation time remains valid. A rejected result changes nothing: it does not
replace the accepted evidence, does not refresh the freshness reference, and
does not advance the recovery streak, so a stale or out-of-order flood cannot
keep a result artificially fresh. No ordering history is retained.

### Reset and epoch lifecycle

`reset()` begins a fresh supervisor epoch: no source-order baseline, no cached
safety evidence, no recovery streak, no hard latch, component health back to
**unknown**, state back to `INITIALIZING`, and `autonomy_allowed = False`.
Because health returns to unknown, the new epoch must observe component health
again before autonomy can be permitted — clearing a latch is not the same as
proving the components recovered. Repeated reset is safe and idempotent.
Lifetime scalar counters, including the reset count, remain truthful, matching
the accepted Mission 7 tracker convention.

The full lifecycle contract for a new autonomous epoch is:

```text
new runtime epoch  ->  fresh or reset temporal pipeline  ->  reset safety supervisor
```

This is required, not advisory, and it matters most **after a faulted runtime
epoch**, where Mission 7's tracker may already have advanced past the last
published result.

### Boundedness, threading, and metrics

The supervisor is single-consumer and lock-free. It creates no thread, queue,
timer, or clock, and performs no clock read of its own: safety time is always
injected. Its retained state is one decision, one canonical accepted source
identity, a small set of scalars, and bounded lifetime counters.

Out-of-repository stress and an independent adversarial review both measured the
retained object graph at a low checkpoint and again after long streams and found
**zero growth with stream length** — the retained set is constant, not merely
slow-growing. Absolute object and byte totals are deliberately not quoted here:
they depend on how the traversal treats shared type, module, and enum objects,
so any single number would be methodology-dependent rather than a property of
the supervisor. The reproducible claim is the constancy.

`SafetySupervisorMetrics` is bounded scalar diagnostics only. **No safety
behavior is derived from any metric**, and in particular Mission 8 never reads
Mission 7's deliberately non-concurrent-atomic `TemporalLaneTracker.metrics`.
Like Mission 7's, these counters are read from the single owning consumer; they
are not claimed to be a concurrently atomic multi-counter snapshot.

### Mission 8 offline validation

Mission 8 adds 115 deterministic tests across two new discoverable files, of
which 21 are R1 regressions for the unknown-health contract above.

The fresh independent adversarial review accepted Mission 8 with **BLOCKER =
0, MAJOR = 0, MINOR = 1** and the verdict **PASS —
MISSION_8_R1_READY_FOR_FINAL_FREEZE_AND_COMMIT**.

| Evidence | Result |
|----------|--------|
| Mission 8 suites | 115/115 pass |
| Critical Mission 5/6/7 collection | 408/408 pass |
| Critical Mission 5-8 superset | 523/523 pass |
| Full offline discovery | 1,301/1,301 pass |
| Static test methods, all files | 1,303 |
| Statically discoverable and executed | 1,301 |
| Structurally excluded live methods | 2 |
| Failures / errors / skips | 0 / 0 / 0 |
| CARLA | not contacted for Mission 8 |

Independent adversarial evidence included 1,080 targeted UNKNOWN-health cases
with zero autonomy leakage and 62,030 randomized operations with zero
safety-invariant violations. The review independently reproduced the F1 and F2
defenses and confirmed O(1) retained state.

The counts reconcile exactly with the Mission 7 baseline of 1,188 static and
1,186 discoverable: all 115 additional methods are Mission 8 tests, and the two
structural exclusions in `tests/integration_perception_runtime.py` and
`tests/integration_sensor_suite.py` are unchanged opt-in live harnesses.

The "critical Mission 5/6/7 collection" is exactly these seven modules, so the
408 is reproducible rather than a bare assertion:

| Module | Tests |
|--------|------:|
| `tests.test_lane_perception` | 183 |
| `tests.test_perception_runtime` | 69 |
| `tests.test_temporal_lane_tracking` | 60 |
| `tests.test_live_camera_snapshot_integration` | 40 |
| `tests.test_live_bridge` | 26 |
| `tests.test_integration_perception_runtime` | 20 |
| `tests.test_integration_temporal_perception_runtime` | 10 |
| **Total** | **408** |

Adding `tests.test_safety_supervisor` (100) and
`tests.test_integration_safety_supervisor` (15) gives the 523 Mission 5-8
superset.

Out-of-repository adversarial stress covered long evaluate/update streams,
duplicate and out-of-order floods that accepted nothing and restored no
autonomy, long no-new-result schedules that permitted exactly the inclusive
freshness window and then stayed fail-safe, soft-loss and recovery cycles with
zero early enables and re-enable on the exact Nth sample, hard-fault and reset
cycles with zero autonomy leaks and no retained exception or traceback object,
long unknown-health streams that never permitted autonomy, and hostile numeric,
stamp, health, and configuration probes, none of which were retained. Retained
state was profiled at a low checkpoint and again at the end of each stress and
showed no growth. Fresh-subprocess import probes of `safety` and
`safety.supervisor` confirmed no `carla`, `numpy`, `pygame`, `cv2`, `scipy`,
`socket`, or `ssl` import, no thread creation, and no audited socket,
subprocess, or network activity.

One accepted test-only MINOR remains in
`test_rising_frame_stream_without_health_never_permits_autonomy` in
`tests/test_safety_supervisor.py`: its floating-point `now` construction can
eventually decrease slightly, latch `CLOCK_REGRESSED`, and therefore make part
of that test exercise the wrong fail-closed reason instead of
`COMPONENT_NOT_READY`. Production behavior is unaffected, and the intended
UNKNOWN-health scenario is independently covered elsewhere. This is accepted
as bounded backlog; no remediation is required for Mission 8 acceptance.

### Mission 8 scope boundary

Mission 8 is **independently reviewed and OFFLINE accepted**. The accepted R1
checkpoint has no BLOCKER or MAJOR findings; the one bounded test-only MINOR is
documented above.

No live CARLA Mission 8 acceptance has been performed, and the accepted Mission
6B live evidence above must not be read as safety-supervision acceptance or as
production-road readiness.

Mission 8 contains no steering, throttle, braking, command-safety envelope,
autopilot, scenario runner, fault injection, logging mission, replay, cinematic
work, dashboard, startup orchestration, or any other vehicle-control behavior.
A safety *decision* exists; a vehicle *command* does not. Missions 10-19 remain
unimplemented.

## Mission 9 - lateral control (offline implementation)

Mission 9 answers exactly one question: given the Mission 7 temporal lane
estimate that Mission 6 published, and the Mission 8 permission issued for that
very same evidence, what bounded lateral steering *intent* may be requested
right now?

It is an intent producer, not an actuator. It builds no `carla.VehicleControl`,
calls no `apply_control`, owns no CARLA actor, and emits no throttle, brake,
target speed, or other longitudinal command. It creates no thread, queue, timer,
socket, or clock: the caller supplies host-monotonic control time explicitly,
from the same clock it gave Mission 8. Nothing Mission 9 produces reaches a
vehicle; turning a request into a command is future Mission 10 work that does
not exist.

The package is `control/`, with `control/lateral.py` holding
`LateralController`, the immutable `LateralControlRequest` it publishes, the
validated `LateralControllerConfig`, the bounded `LateralControlMode` and
`LateralControlReason` enumerations, and `LateralControllerMetrics`. No CARLA
type appears anywhere in that public API.

### Consumed evidence

`LateralController.update(result, decision, now=...)` consumes one published
`PerceptionRuntimeResult` carrying a Mission 7 `TemporalLaneObservation`, plus
the Mission 8 `SafetyDecision` issued for that result, plus explicit control
time. Both records are read and never retained: everything Mission 9 keeps is
copied into exact built-in scalars or a freshly constructed `SampleStamp`.

### The Mission 8 safety gate

Mission 8 remains the sole authority on whether autonomy is permitted. Mission 9
never re-decides that question and can only *subtract* authority, never add it.
The gate is evaluated before any steering value is computed at all, and a
request is refused unless all three of `autonomy_allowed is True`, a state of
`NOMINAL` or `DEGRADED`, and `latched is False` hold together, each attribute
read exactly once so a hostile subclass cannot answer one value to the gate and
another to the control law. Perfect lane geometry cannot bypass it. Every
refusal publishes `lateral_allowed=False` and exactly `0.0` steering.

### Evidence identity binding

A decision for frame N must never authorize control using frame M, so Mission 9
proves the pairing for itself rather than trusting that the caller kept them
together. It repeats the accepted Mission 8 outer/carrier/estimate binding
(runtime `source_stamp` and `snapshot_revision` against the raw carrier's stamp
and revision, against the temporal estimate's stamp and revision, and the raw
detection state the carrier and estimate must agree on), and then adds the
Mission 9-specific decision binding: the decision's `source_stamp`,
`snapshot_revision`, `temporal_state`, and `temporal_confidence` must all equal
the evidence actually supplied. Any mismatch is refused with
`IDENTITY_MISMATCH` and zero steering.

Reading published evidence is pure attribute access and validation. An accepted
*exact* Mission 8 `SafetyDecision` is a frozen, slotted dataclass, so every field
read Mission 9 performs on it is a stable slot read that cannot raise. The two
evidence reads that a hostile object could reach - the Mission 6 result structure
together with the outer/carrier/estimate identity, and the decision identity
fields `source_stamp`, `snapshot_revision`, `temporal_state`, and
`temporal_confidence` - are wrapped, and any exception from them fails closed as
`EVIDENCE_MALFORMED` rather than escaping to the caller.

The three Mission 8 gate reads themselves - `autonomy_allowed`, `state`, and
`latched` - are **not** wrapped. A hostile `SafetyDecision` *subclass* whose
descriptors for those three raise is outside the accepted pipeline, and such an
exception may propagate to the caller instead of being converted into a refusal.
It cannot pass the safety gate and cannot produce a newly permitted steering
request, because the gate never completes and no new request is published; the
previously published request may remain observable on `latest_request`. This is
an accepted bounded backlog item, and it is deliberately *not* a claim that every
exception from every hostile attribute read is caught.

Mission 8 owns permanent safety latching;
Mission 9 adds none, because a refusal here is already fail-closed for that
evaluation and a genuine hard fault latches upstream.

Mission 9 additionally keeps one canonical accepted source stamp and applies the
accepted Mission 6 order semantics to it: same frame *and* same simulation time
is a duplicate, which is a legitimate re-evaluation of the newest evidence and
is accepted; a lower frame, an equal frame with a different simulation time, or
a regressed simulation time is refused as `SOURCE_REJECTED` without moving that
baseline. This state is not redundant with the identity binding: an internally
coherent (result, decision) pair from an older frame can be replayed later, and
only this baseline refuses it.

### Geometry source

Guidance comes from the Mission 7 *temporal* estimate, never from the raw
Mission 5 observation, through one narrow extraction function. A `COASTING`
estimate whose current raw frame detected nothing therefore still yields bounded
extrapolated guidance, while an unusable raw frame never reaches the control
law. Mission 7 exposes complete `center_offset`/`heading_proxy` only when both
boundaries are present, so a partial track has no lane centre and Mission 9
refuses with `INSUFFICIENT_GEOMETRY` rather than inventing the missing side.
`LOST`, `INPUT_UNUSABLE`, and `UNINITIALIZED` refuse the same way. Mission 8
saying "degraded autonomy is permitted" - which it really does say for partial
tracking - does not force Mission 9 to manufacture geometry it does not have.

### Steering sign convention

Positive requested steering means "steer toward increasing normalized image x",
that is, toward the right-hand side of the camera image. Negative means left,
zero means straight ahead. The value is a normalized, dimensionless intent
bounded to `[-1.0, +1.0]`; it is **not** a road-wheel angle, and it is not world
lateral error, vehicle heading, pose, or world-metric lane geometry.

The convention is derived from the published Mission 5 definitions documented
above, not assumed. `center_offset` is `lane_center_x - 0.5` at normalized image
row `0.90`, so positive means the lane centre lies to the right of the image
centre near the ego. `heading_proxy` is the image-space centreline slope with y
increasing downward, so positive means the centreline moves rightward toward the
bottom of the image, equivalently leftward as rows move up toward road further
ahead. Because Mission 5 fits straight boundaries, the centreline is straight and
its column at any row is

```
x_c(y) = (0.5 + center_offset) + heading_proxy * (y - 0.90)
```

Evaluating that line a normalized lookahead `L` further ahead, i.e. `L` rows
above the evaluation row, gives the lane-centre displacement from the image
centre at the lookahead row:

```
x_c(0.90 - L) - 0.5 = center_offset - L * heading_proxy
```

That is positive exactly when the lane centre at the lookahead row lies right of
the image centre, which is exactly when the ego must steer right to move toward
the lane centre.

### Equation and normalization

```
raw_steering = offset_gain * center_offset - heading_gain * heading_proxy
```

The ratio

```
L = heading_gain / offset_gain
```

is what the control law actually applies. Interpreting `L` as a lookahead row
*inside the rows Mission 5 actually fitted* is valid only when

```
0 <= L <= center_evaluation_y - roi_top = 0.90 - 0.50 = 0.40
```

using the default Mission 5 `center_evaluation_y = 0.90` and `roi_top = 0.50`.
The default gains satisfy it: `L = 0.40 / 1.20 = 1/3` gives an effective
lookahead row of `0.90 - 1/3 = 0.5667...`, which lies inside the default Mission
5 ROI `[0.50, 1.00]`.

Configuration validation bounds `offset_gain` and `heading_gain` independently
and places no constraint on their ratio, so an accepted configuration may set
`L > 0.40`. Such a ratio evaluates the fitted straight centreline above the ROI
Mission 5 fitted, so it is extrapolation rather than a validated in-ROI
lookahead; the request remains bounded by the steering authority clamp and the
slew limit, but the in-ROI lookahead interpretation no longer holds. This is an
accepted bounded backlog item.

The mathematics is deterministic scalar arithmetic: no machine learning,
optimizer, NumPy, SciPy, or adaptive tuning is involved.

No image-width or pixel normalization constant appears anywhere. Both consumed
scalars are already normalized and bounded to `[-1.0, 1.0]` by the accepted
Mission 5 contract, so there is nothing to rescale and no image size to
hardcode, and no `offset_normalization_scale` setting exists.

### Nominal versus degraded authority

`NOMINAL` authority requires a Mission 8 `NOMINAL` state *and* a live,
non-extrapolated `TRACKING` estimate. Everything else that is still permitted is
`DEGRADED`, reported as `DEGRADED_COASTING` for an extrapolated track and
`DEGRADED_TRACKING` otherwise. Coasting can therefore never obtain nominal
authority even if Mission 8 were to call it nominal. Degraded authority is a
strictly lower configured bound, so the same large error produces a lower or
equal absolute request under degradation.

### Clamping, rate limiting, and time

| Setting | Default | Meaning |
|---------|--------:|---------|
| `offset_gain` | 1.20 | multiplier on the normalized centre offset |
| `heading_gain` | 0.40 | multiplier on the normalized heading proxy |
| `nominal_max_steering` | 0.35 | authority for fully healthy evidence |
| `degraded_max_steering` | 0.15 | authority for degraded evidence |
| `max_steering_rate_per_second` | 1.50 | slew bound per real second |
| `steering_deadband` | 0.0 | inert by default |

Validation requires `0 < degraded_max_steering <= nominal_max_steering <= 1.0`,
a positive bounded rate, a positive bounded offset gain, a non-negative bounded
heading gain, and a deadband in `[0, degraded_max_steering)`. NaN, both
infinities, `bool`-as-number, non-real values, and hostile `numbers.Real`
objects whose value changes between reads are rejected or canonicalized to exact
built-in floats, and the caller's configuration object is never retained.

The raw request passes through the deadband, is clamped to the current
authority, is then slew limited, and the result is bounded by the authority
again. The slew anchor is the previously published steering brought inside the
*current* authority first, so a nominal-to-degraded downgrade reduces magnitude
immediately rather than slewing while still exceeding the lower bound; both
anchor and target then lie inside the authority, so the rate-limited value does
too. `abs(steering) <= steering_authority` holds unconditionally.

Mission 9 owns no clock. The caller supplies `now`, the elapsed time is
`now - previous_evaluation_time`, and the first command in an epoch has no
previous value and is bounded by the authority clamp alone. Zero elapsed time
permits no change at all, a tiny elapsed time permits a proportionally tiny
change, and a long elapsed time reaches but never overshoots the target. A
non-finite or regressed control clock fails closed with `TIME_INVALID` and
discards the slew baseline entirely, because no slew statement can be made
across an untrustworthy clock.

The deadband is a separate shaping step from the authority clamp, and
suppressing a tiny request is not counted or reported as a clamp activation.

A refusal publishes exactly zero and that zero is deliberately not rate limited:
reducing intent to neutral is always permitted, and a safety refusal must not be
slewed in over several control steps. The published zero becomes the new slew
anchor, so re-enabling ramps up from neutral rather than resuming a stale
command, and a refused or replayed observation can never inflate a later slew.
An accepted duplicate does advance the slew clock, which is correct: the bound
is on real actuator slew per real second, and re-evaluating unchanged evidence
leaves the target unchanged, so the ramp only continues toward the same value it
was already approaching.

### Reset and lifecycle

`reset()` clears the previous steering value, the rate-limit time, and the
retained source baseline, and republishes a neutral, refused `NOT_EVALUATED`
request. No previous command crosses an epoch boundary, and repeated resets are
idempotent apart from the truthful lifetime counter. A new runtime epoch should
reset or recreate Mission 7, then Mission 8, then Mission 9.

### Boundedness and concurrency

Retained state is O(1): one immutable request, one previous steering value and
its evaluation time, one canonical accepted source stamp, one detached config,
and bounded lifetime counters. There is no command, frame, or decision history,
no queue, and no log. The controller is single-consumer and lock-free; its
metrics are diagnostics read by that same consumer and are explicitly **not**
claimed to be concurrently atomic.

### Mission 9 offline validation

Mission 9 adds 109 deterministic tests across two new discoverable files:
98 unit tests in `tests/test_lateral_control.py` and 11 real-pipeline
integration tests in `tests/test_integration_lateral_control.py`.

| Evidence | Result |
|----------|--------|
| Mission 9 suites | 109/109 pass |
| Critical Mission 5-9 collection | 632/632 pass |
| Full offline discovery | 1,410/1,410 pass |
| Static test methods, all files | 1,412 |
| Statically discoverable and executed | 1,410 |
| Structurally excluded live methods | 2 |
| Failures / errors / skips | 0 / 0 / 0 |
| CARLA | not contacted for Mission 9 |

The counts reconcile exactly with the Mission 8 baseline of 1,303 static and
1,301 discoverable: all 109 additional methods are Mission 9 tests, and the two
structural exclusions in `tests/integration_perception_runtime.py` and
`tests/integration_sensor_suite.py` are unchanged opt-in live harnesses. The
critical collection is the accepted Mission 5-8 superset of 523 plus the 109 new
Mission 9 methods.

The integration suite drives the accepted Mission 6A runtime, Mission 7 temporal
pipeline, and Mission 8 supervisor for real, injecting only a Mission 5-shaped
perception callable and explicit clocks. It proves real nominal tracking
steering, a near-zero request for a centred lane, exactly mirrored steering for
mirrored displacement and for mirrored heading evidence, reduced authority under
real Mission 7 coasting with the raw frame carrying no geometry at all, zero
steering under real `FAIL_SAFE`, `RECOVERING`, `LOST`, and `INPUT_UNUSABLE`,
zero steering when a genuine Mission 8 decision for an older frame is paired
with a newer result, a refusal to steer on real partial evidence that Mission 8
genuinely permits as degraded, and a new epoch in which no previous command or
slew baseline survives.

Out-of-repository stress covered 250,000 valid updates, 100,000 safety-disabled
inputs, 100,000 alternating extreme errors, 50,000 resets, 100,000 identity
mismatches, and 64 hostile numeric and configuration probes. No refused input
ever produced a permitted request or a nonzero steering value; the authority and
slew envelopes were never exceeded, with a worst measured overshoot of exactly
`0.0` on both; every hostile probe was rejected or canonicalized and no caller
object was retained. Retained state was profiled at checkpoints throughout and
varied by under one kilobyte in total, and the per-call cost of `update()`
stayed inside a narrow band across the whole 250,000-update stream (about 14
microseconds per call when the allocation tracer is not attached).

A fresh-subprocess import probe of `control` and `control.lateral` confirmed no
`carla`, `numpy`, `pygame`, `cv2`, `scipy`, `socket`, `subprocess`, `asyncio`,
`multiprocessing`, or networking import, and no thread creation. `threading` is
reachable only transitively through the accepted Mission 6 runtime module;
`control.lateral` itself imports nothing outside `__future__`, `math`,
`numbers`, `dataclasses`, `enum`, and the accepted `perception`, `safety`, and
`telemetry` packages, and a test asserts that import allowlist directly.
Identifier-level checks over the module confirm no `apply_control`,
`VehicleControl`, `throttle`, `brake`, `autopilot`, or CARLA actor name exists in
the implementation.

### Mission 9 scope boundary

**Mission 9 offline implementation passed a fresh independent adversarial review
with BLOCKER = 0 and MAJOR = 0, and is accepted for final freeze and local
commit.** The review returned two accepted bounded MINOR items, both documented
in full above: a hostile `SafetyDecision` subclass whose `autonomy_allowed`,
`state`, or `latched` descriptor raises may propagate that exception rather than
be converted into a refusal, and a configured gain ratio above `0.40`
extrapolates beyond the Mission 5 ROI instead of being an in-ROI lookahead.
Neither can pass the safety gate or exceed the steering authority bound.

No live CARLA Mission 9 acceptance has been performed, and the accepted Mission
6B live evidence above must not be read as lateral-control acceptance or as
production-road readiness.

No steering is applied to a vehicle. There is no throttle, brake, longitudinal
control, target-speed control, actuator command, command-safety envelope,
autopilot, scenario runner, fault injection, dashboard, or startup
orchestration in Mission 9. A lateral *request* exists; a vehicle *command* does
not. Missions 10-19 remain unimplemented.
