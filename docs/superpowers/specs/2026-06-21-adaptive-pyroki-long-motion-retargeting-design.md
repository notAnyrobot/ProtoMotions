# Adaptive PyRoki Long-Motion Retargeting Design

## Status

Design approved by discussion; pending written-spec review.

## Problem

The AMASS train split contains long motions that are too large for the current
PyRoki batch solver to optimize as one JAXLS problem. A concrete failure was a
5915-frame keypoint motion, which produced a very large problem and failed in
JAXLS analysis with an integer overflow.

Truncating with `--target-raw-frames` is not acceptable because tracker policy
training should retain the full trajectory represented by the source AMASS clip.
Splitting long motions into multiple final MotionLib entries is also not the
desired default because it changes dataset semantics and sampling distribution.

## Goals

- Preserve one final robot motion per original AMASS motion.
- Retarget full-length long motions without skipping or truncating frames.
- Keep existing behavior for normal-length motions.
- Keep the final file layout unchanged:
  - one input `*_keypoints.npy`
  - one output `*_retargeted.npz`
  - one converted `.motion`
  - one packaged MotionLib entry
- Make interrupted HPC runs resumable with `--skip-existing`.
- Keep converter and packager behavior independent of chunk internals.

## Non-Goals

- Do not change AMASS split YAMLs.
- Do not create multiple final MotionLib entries for one source motion by default.
- Do not solve cross-motion blending. Stitching applies only to chunks from the
  same source motion.
- Do not retune robot-specific PyRoki weights as part of this change.
- Do not change the default output FPS policy in the converter.

## Recommended Design

Add adaptive chunking inside the PyRoki retargeting solver boundary.

The batch retargeter still iterates over keypoint `.npy` files. For each file:

1. Load full keypoint and contact data as today.
2. If the motion length is at or below `chunk_threshold_frames`, run the current
   full-motion solve path unchanged.
3. If the motion length exceeds `chunk_threshold_frames`, split the loaded
   motion into overlapping windows.
4. Retarget each window independently with the existing solver.
5. Stitch the retargeted chunk outputs into one full-length robot trajectory.
6. Save one `*_retargeted.npz` at the existing output path.

The converter and MotionLib packager continue to read only the stitched final
`.npz` files. They do not need to know that a long motion was internally chunked.

## Default Parameters

Initial defaults:

```text
chunk_threshold_frames = 900
chunk_size_frames = 450
chunk_overlap_frames = 60
```

Meaning:

- Motions of 900 frames or shorter use the existing full-motion solve.
- Motions longer than 900 frames are solved in windows of up to 450 frames.
- Neighboring windows overlap by 60 frames.

The threshold is intentionally larger than the chunk size. This avoids chunking
moderately sized motions that are still likely to solve cleanly, while keeping
large windows near the previous practical cap.

## CLI Surface

Add PyRoki CLI options:

```text
--chunk-long-motions
--chunk-threshold-frames 900
--chunk-size-frames 450
--chunk-overlap-frames 60
```

Behavior:

- `--chunk-long-motions` enables adaptive chunking.
- If not enabled, behavior remains exactly as today.
- `--target-raw-frames` remains a truncation/debug option. It should not be used
  for production full-length AMASS retargeting when chunking is enabled.
- The HPC convenience script should pass `--chunk-long-motions` for all splits.
  It is a no-op for motions at or below the threshold.

Validation rules:

- `chunk_size_frames > 0`
- `chunk_overlap_frames >= 0`
- `chunk_overlap_frames < chunk_size_frames`
- `chunk_threshold_frames >= chunk_size_frames`

## Window Generation

For a motion with `T` frames:

1. If `T <= chunk_threshold_frames`, solve `[0, T)`.
2. Else use stride:

```text
stride = chunk_size_frames - chunk_overlap_frames
```

3. Generate windows:

```text
[0, chunk_size)
[stride, stride + chunk_size)
[2 * stride, 2 * stride + chunk_size)
...
```

4. Clamp each window end to `T`.
5. Ensure the last window reaches frame `T`.
6. Avoid tiny final windows by shifting the final start backward when needed so
   the final window has useful context, while keeping overlap with the previous
   window.

Example for `T = 5915`, `chunk_size = 450`, `overlap = 60`:

```text
stride = 390
windows roughly: [0,450), [390,840), [780,1230), ... final window ending at 5915
```

## Stitching Contract

Each chunk solve returns:

```text
base_frame_pos:  [chunk_T, 3]
base_frame_wxyz: [chunk_T, 4]
joint_angles:   [chunk_T, num_dof]
```

The stitched result must have exactly:

```text
base_frame_pos:  [T, 3]
base_frame_wxyz: [T, 4]
joint_angles:   [T, num_dof]
```

For non-overlap regions, copy chunk outputs directly.

For overlap regions, blend the previous stitched output with the new chunk:

- Root position: linear blend.
- Root rotation: quaternion SLERP in WXYZ order with sign correction.
- Joint angles: unwrap per joint before blending, then store continuous angles.

Blend weights across an overlap of length `N`:

```text
alpha = linspace(0.0, 1.0, N)
stitched = (1 - alpha) * previous + alpha * current
```

Use endpoint-inclusive weights so the first overlap frame favors the previous
chunk and the last overlap frame favors the current chunk.

## Coordinate Continuity

Chunk solves should use source keypoint root positions as usual, so global
translation remains anchored to the original motion. The stitching layer should
still check for large discontinuities at chunk boundaries:

- root position jump
- root rotation angular jump
- max per-joint angle jump

These diagnostics should be printed for long motions. They are warning signals,
not automatic failures in the first implementation.

## Contact Labels

Contacts should remain source-derived, as in the current pipeline.

Keep contacts-only extraction independent from retarget chunking. It reads the
original full keypoint `.npy` and writes one full-length contact `.npz`.
Contacts do not require optimization and should not be windowed.

The output contact file must remain one `*_contacts.npz` per source motion.

## FPS Handling

Keypoint `.npy` files do not store FPS. The PyRoki solver uses `--input-fps` for
velocity-related costs. The converter uses `--input-fps` and `--output-fps` for
resampling and saved `.motion` timing.

The chunking implementation must not infer FPS from frame counts. It should use
the existing `input_fps` option and pass that value consistently to each chunk
solve. For the first implementation, the HPC script should expose an explicit
`PROTO_INPUT_FPS` override and pass it to PyRoki. Automatic per-motion FPS
metadata propagation is useful but out of scope for this chunking change.

This is important because local AMASS MotionLibs can contain 50 Hz and 60 Hz
motions.

## Resume Behavior

The default saved artifact remains one `*_retargeted.npz`. Therefore current
`--skip-existing` semantics stay simple:

- If final `*_retargeted.npz` exists, skip the motion.
- If it does not exist, retarget the full source motion, chunking internally when
  needed.

Optional future improvement:

- Save temporary per-chunk files under an internal scratch directory and resume
  at chunk granularity. This is not required for the first implementation.

## Failure Behavior

If one chunk fails:

- Do not write the final stitched `*_retargeted.npz`.
- Print the source keypoint file, chunk index, frame range, and exception.
- Exit non-zero in strict mode so the HPC job surfaces the failure.

Optional future improvement:

- Add a failure-tolerant mode that logs failed motions and continues with the
  next source motion. This is useful for production sweeps but should be a
  separate policy decision from chunking.

## Integration Points

Primary code paths:

- `pyroki/retargeting/cli.py`
  - Add chunking CLI options to `BatchRetargetingOptions`.
- `pyroki/retargeting/solver.py`
  - Add window generation helpers.
  - Add chunked solve orchestration.
  - Add output stitching helpers.
  - Use chunked solve only when enabled and over threshold.
- `scripts/docker/retarget_amass_isaaclab_hpc.sh`
  - Pass chunking options for production retargeting.
  - Expose environment overrides for threshold, chunk size, and overlap.
- `scripts/docker/README.md`
  - Document long-motion retargeting defaults and when to adjust them.

No changes should be required in:

- `data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py`
- `protomotions/components/motion_lib.py`

## Testing Plan

Unit tests:

- Window generation covers short, exact-threshold, long, and final-window cases.
- Invalid chunk parameter combinations are rejected.
- Stitching preserves exact output length.
- Stitching copies non-overlap frames unchanged.
- Stitching blends overlap frames for position and joints.
- Quaternion stitching handles sign-corrected WXYZ quaternions.

CLI tests:

- New options parse into `BatchRetargetingOptions`.
- Defaults preserve current behavior when `--chunk-long-motions` is absent.

Batch retargeting tests with fakes:

- Short fake motion calls the existing single-solve path.
- Long fake motion calls the solver once per window and writes one final `.npz`.
- `--skip-existing` still skips an existing final `.npz`.

HPC script tests:

- Script forwards chunking flags to the PyRoki retarget step.
- Script does not pass chunking flags to contacts-only extraction, because
  contacts remain full-length and unchunked.
- Script forwards `PROTO_INPUT_FPS` to the PyRoki retarget step when set.

Manual validation:

- Retarget one known long train motion, including the 5915-frame failure case.
- Convert to `.motion` and package.
- Inspect the stitched motion in `examples/motion_libs_visualizer.py`.
- Check boundary diagnostics for the long motion.
- Confirm the packaged MotionLib contains one entry for the source motion.

## Rollout Plan

1. Implement and test chunk helpers with pure NumPy fake data.
2. Add CLI options with default behavior unchanged.
3. Route long motions through chunked solve when enabled.
4. Add stitching diagnostics.
5. Update the HPC convenience script and README.
6. Validate on the previously failing 5915-frame train motion.
7. Resume full train retargeting with `--skip-existing`.

## Decisions For First Implementation

- Enable adaptive chunking in the HPC helper for all splits.
- Keep contacts-only extraction full-length and unchunked.
- Add explicit `PROTO_INPUT_FPS` support in the HPC helper, but do not implement
  automatic per-motion FPS propagation in this change.
- Do not add chunk-level scratch outputs in the first implementation. Resume at
  the final `*_retargeted.npz` artifact level.
