# Grid Motion Safety

This is a minimal RoboTwin validation task for checking whether a floating
visual-only grid marker affects robot motion.

The task intentionally allows the camera to see the grid. The validation target
is motion safety:

- markers are visual-only SAPIEN entities;
- markers use high-contrast `RenderBodyComponent` with `RenderShapeSphere`
  points plus `RenderShapeBox` grid lines;
- markers do not add PhysX rigid bodies or collision shapes;
- pointcloud collection is disabled in the clean config;
- the grid center, grid size, arm mode, and selected target points are controlled
  by constants at the top of `grid_motion_safety_core.py`;
- the left TCP, right TCP, or both TCPs can move to selected 3x3x3 grid points;
- dynamic visual-only TCP reference markers follow the current left/right TCPs;
- camera frames are sampled every 10 control steps inside this task to keep the
  one-episode validation small enough to merge into HDF5/video reliably.

## Motion Controls

Edit the constants near the top of `grid_motion_safety_core.py`:

```python
GRID_CENTER_MODE = "arms_midpoint"  # "arms_midpoint", "left_tcp_offset", or "absolute"
GRID_CENTER_OFFSET_XYZ = [-0.035, 0.10, 0.08]
GRID_SPACING_XYZ = [0.07, 0.07, 0.07]

ARM_MOTION_MODE = "left"  # "left", "right", "both_same", or "both_mirrored"
GRID_TARGET_POINT_IDS = [13]  # Use "all" to visit all 27 grid points.
DUAL_MIRROR_AXIS = "x"
```

`GRID_SPACING_XYZ` is the half-span around the grid center.  The current 3x3x3
point ids are generated as `pos_id = iz * 9 + iy * 3 + ix`, where each index is
`0, 1, 2`; the center point is `13`.

`both_mirrored` sends the left arm to the selected grid point and the right arm
to the mirrored point around the grid center along `DUAL_MIRROR_AXIS`.

## Files

- External core: `rpy_direction_validation/grid_motion_safety_core.py`
- RoboTwin shim: `third_party/robotwin/envs/rpy_grid_motion_safety.py`
- Config: `third_party/robotwin/task_config/rpy_grid_motion_safety_clean.yml`
- Fixed report: `rpy_direction_validation/grid_motion_safety_report.json`
- Data report: `third_party/robotwin/data/rpy_grid_motion_safety/rpy_grid_motion_safety_clean/motion_safety_report.json`

## Run

From the RoboTwin directory:

```bash
bash collect_data.sh rpy_grid_motion_safety rpy_grid_motion_safety_clean 0 --denoiser none
```

On this machine, RoboTwin may also require:

```bash
PATH=/data1/liuwenhao/conda/envs/RoboTwin/bin:$PATH
MPLCONFIGDIR=/data1/liuwenhao/tmp/mplconfig
XDG_CACHE_HOME=/data1/liuwenhao/tmp/xdg-cache
TORCH_CUDA_ARCH_LIST=12.0
LD_LIBRARY_PATH=/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library:$LD_LIBRARY_PATH
```

## Expected Result

`grid_motion_safety_report.json` should have:

```json
{
  "conclusion": "pass",
  "marker": {
    "has_collision": false,
    "uses_add_collision_api": false
  },
  "motion_test": {
    "all_moves_success": true
  }
}
```

If planning fails, shrink the grid path or raise the grid center slightly before
using this task as a camera visibility calibration base.
