# Viewer-only Debug Geometry Investigation

## 1. Environment

- RoboTwin path: `/data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin`
- Python executable: `/data1/liuwenhao/conda/envs/RoboTwin/bin/python`
- SAPIEN version: `3.0.0b1`
- SAPIEN package path: `/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien`
- Note: importing SAPIEN with the raw RoboTwin env fails with `ImportError: libOpenImageDenoise.so.2`. The same import succeeds when using the runtime library path used by RoboTwin collection:

```bash
LD_LIBRARY_PATH=/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library:$LD_LIBRARY_PATH
```

## 2. Search Summary

Searched RoboTwin keywords:

```text
debug, viewer, draw, line, point, sphere, visual, render, visibility,
visible, mask, camera_mask, render_mask, hide
```

Main RoboTwin hits:

- `envs/_base_task.py`
  - imports `Viewer` from `sapien.utils.viewer`
  - creates `self.viewer = Viewer(self.renderer)` only when `render_freq` is enabled
  - calls `self.viewer.set_scene(self.scene)`
  - calls `self.scene.update_render()` before camera capture
- `envs/camera/camera.py`
  - creates cameras through `scene.add_camera(...)`
  - captures saved RGB through `camera.take_picture()` and `camera.get_picture("Color")`
- `envs/utils/create_actor.py`
  - creates normal visual entities with `sapien.render.RenderBodyComponent`
  - helper shapes include `RenderShapeBox`, `RenderShapeSphere`, `RenderShapeCylinder`
- `envs/utils/transforms.py`
  - `add_robot_visual_box(...)` creates a normal scene actor through `scene.create_actor_builder()`
- `script/create_object_data.py` and `script/create_messy_data.py`
  - contain visual helper code, but these are normal scene/viewer workflows, not camera-hidden debug geometry

Searched SAPIEN package keywords:

```text
debug, draw, line, point, mask, visibility, visible, hide,
camera_mask, render_mask, render_group, visibility_group
```

Main SAPIEN hits:

- `sapien/utils/viewer/viewer.py`
  - `Viewer.add_bounding_box(...)`
  - `Viewer.draw_aabb(...)`
  - uses `self.renderer_context.create_line_set(...)`
  - adds it to `self.render_scene.add_line_set(...)`
- `sapien/pysapien/internal_renderer.pyi`
  - `Context.create_line_set(...)`
  - `Context.create_point_set(...)`
  - `Scene.add_line_set(...)`
  - `Scene.add_point_set(...)`
  - `Scene.remove_node(...)`
- `sapien/pysapien/render.pyi`
  - `RenderBodyComponent.visibility`
  - `RenderBodyComponent.disable_render_id()`
  - `RenderBodyComponent.enable_render_id()`
  - `RenderCameraComponent` has no mask / visibility-group API in the stub

Exact `camera_mask` and `render_mask` searches did not find usable APIs in RoboTwin or SAPIEN.

## 3. Candidate APIs Found

### Candidate 1: `sapien.utils.viewer.Viewer.draw_aabb`

- API / function / class name: `Viewer.draw_aabb(lower, upper, color)`
- File path: `/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/utils/viewer/viewer.py`
- Relevant code:

```python
def draw_aabb(self, lower, upper, color):
    pose = sapien.Pose((lower + upper) / 2)
    half_size = (upper - lower) / 2
    return self.add_bounding_box(pose, half_size, color)
```

- Underlying implementation:

```python
lineset = self.renderer_context.create_line_set(vertices, colors)
box = self.render_scene.add_line_set(lineset)
```

- Possible use: viewer-only debug wire boxes / grid cells / line visualizations.
- Whether viewer-only: yes, based on code path. It adds to `Viewer.render_scene`, which is `RenderWindow._internal_scene`, not to the RoboTwin simulation `sapien.Scene`.
- Whether it enters camera rendering: unlikely / no by code structure. RoboTwin saved RGB uses cameras from `self.scene.add_camera(...)`, then `scene.update_render()`, `camera.take_picture()`, and `camera.get_picture("Color")`. `Viewer.render_scene` line nodes are not added to `self.scene`.
- Confidence: high for viewer window only; medium-high overall because I did not run a rendered pixel comparison due headless Vulkan renderer initialization failing outside the normal collection runtime.

### Candidate 2: `internal_renderer` line and point sets

- API / function / class name:
  - `Context.create_line_set(vertices, colors)`
  - `Context.create_point_set(vertices, colors)`
  - `Scene.add_line_set(...)`
  - `Scene.add_point_set(...)`
  - `Scene.remove_node(...)`
- File path: `/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/pysapien/internal_renderer.pyi`
- Relevant code:

```python
class Context:
    def create_line_set(self, vertices, colors) -> PointSet: ...
    def create_point_set(self, vertices, colors) -> LineSet: ...

class Scene:
    def add_line_set(self, line_set, parent=None) -> LineSetObject: ...
    def add_point_set(self, point_set, parent=None) -> PointSetObject: ...
    def remove_node(self, node) -> None: ...
```

- Possible use: custom viewer-only line grid / points by following the same pattern as `Viewer.add_bounding_box`.
- Whether viewer-only: yes if added to `viewer.render_scene`, not to `task.scene`.
- Whether it enters camera rendering: no evidence that it would enter RoboTwin camera rendering when attached only to `viewer.render_scene`.
- Confidence: medium-high. The API is used internally by SAPIEN viewer plugins for camera frustums and curves, but RoboTwin does not currently wrap it.

### Candidate 3: `RenderBodyComponent.visibility`

- API / function / class name: `sapien.render.RenderBodyComponent.visibility`
- File path: `/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/pysapien/render.pyi`
- Related viewer use:

```python
if isinstance(c, sapien.render.RenderBodyComponent):
    c.visibility = self.selected_entity_visibility
```

- Possible use: changing normal scene actor opacity / visibility.
- Whether viewer-only: no.
- Whether it enters camera rendering: likely yes, because it belongs to a `RenderBodyComponent` attached to an entity in the simulation scene.
- Confidence: high that it is not a camera-exclusion mechanism. No per-camera filtering behavior was found.

### Candidate 4: `RenderBodyComponent.disable_render_id`

- API / function / class name:
  - `RenderBodyComponent.disable_render_id()`
  - `RenderBodyComponent.enable_render_id()`
  - `RenderBodyComponent.is_render_id_disabled`
- File path: `/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/pysapien/render.pyi`
- Possible use: disabling render id / segmentation id output.
- Whether viewer-only: no.
- Whether it enters camera rendering: it does not remove RGB rendering by name or documented type stub. It appears related to render id / segmentation, not color visibility.
- Confidence: medium. It should not be treated as camera-invisible marker support without a pixel-level experiment.

## 4. Render Mask / Visibility Group Support

No usable render mask / camera mask / visibility group API was found in the current RoboTwin or SAPIEN Python interface.

Negative evidence:

- `camera_mask`: no useful hits.
- `render_mask`: no useful hits.
- `visibility_group`: no useful hits.
- `RenderCameraComponent` type stub contains camera intrinsics, picture capture, shader properties, texture access, and pose methods, but no object inclusion/exclusion mask.
- `RenderBodyComponent` has `visibility`, but no camera-specific mask.

Conclusion for marker actors:

- A normal marker actor created via `RenderBodyComponent` should be assumed to appear in head/wrist/third-view camera RGB if it is in view.
- I did not find a supported way to make a normal scene actor visible only in GUI viewer while hidden from camera rendering.

## 5. Existing Marker Usage in RoboTwin

Existing marker-like code is not camera-safe by default:

- `envs/utils/create_actor.py:create_visual_box(...)`
  - creates a `sapien.Entity`
  - attaches `sapien.render.RenderBodyComponent`
  - adds it to `scene.add_entity(entity)`
  - classification: real visual scene entity; would enter camera rendering if visible.

- `envs/utils/transforms.py:add_robot_visual_box(...)`
  - uses `scene.create_actor_builder()`
  - calls `builder.add_visual_from_file(...)`
  - calls `builder.build()`
  - classification: real scene actor; would enter camera rendering if visible.

- `script/create_object_data.py` / `script/create_messy_data.py`
  - use visual boxes and viewer rendering for annotation/debug tooling
  - classification: normal scene visualization, not proven camera-hidden.

SAPIEN viewer itself uses viewer-internal line sets:

- `sapien/utils/viewer/control_window.py`
  - creates camera frustum lines with `renderer_context.create_line_set(...)`
  - adds them to `viewer.render_scene.add_line_set(...)`
- `sapien/utils/viewer/path_window.py`
  - displays path curves through `viewer.render_scene.add_line_set(...)`

These viewer plugin overlays are not RoboTwin scene actors.

## 6. Conclusion

A. Found reliable viewer-only debug geometry for line-based debug visualization.

The reliable path is to add line/point primitives to `task.viewer.render_scene` through SAPIEN's viewer-internal `internal_renderer` APIs, following the existing `Viewer.add_bounding_box()` / `Viewer.draw_aabb()` pattern. These nodes live in `RenderWindow._internal_scene`, not in RoboTwin's simulation `sapien.Scene`, so they should not be captured by head camera / wrist camera / third view RGB.

Important limitation:

- This conclusion applies to viewer-internal line/point primitives.
- It does not apply to ordinary SAPIEN actors, `RenderBodyComponent`, `create_visual_box`, or `add_robot_visual_box`.
- No camera mask / render mask API was found for normal scene actors.

## 7. Recommended Next Step

Because the conclusion is A, the next step should be a minimal viewer-only test script, not a full data collection run.

Minimal test design:

1. Start a tiny SAPIEN scene with one camera and one visible reference object.
2. Create a `Viewer` and call `viewer.set_scene(scene)`.
3. Add a conspicuous viewer-only line grid with:

```python
lineset = viewer.renderer_context.create_line_set(vertices, colors)
node = viewer.render_scene.add_line_set(lineset)
```

or use:

```python
node = viewer.draw_aabb(lower, upper, color)
```

4. Capture camera RGB with:

```python
scene.update_render()
camera.take_picture()
rgb = camera.get_picture("Color")
```

5. Save one screenshot from the viewer and one camera RGB frame.
6. Verify:
   - viewer screenshot contains the debug lines;
   - camera RGB does not contain the debug lines.

If this pixel-level check passes, RPY/visibility experiments can safely use viewer-internal line sets for interactive debugging. For saved videos, use post-processing overlay if the marker must appear in the saved output without affecting raw camera observations.
