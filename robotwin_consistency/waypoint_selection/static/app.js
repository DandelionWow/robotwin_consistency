import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";

let sceneData = null;
let objectAsset = null;
let threeRenderer = null;
let threeScene = null;
let threeCamera = null;
let threeControls = null;
let gltfLoader = null;
let objectRoot = null;
let environmentRoot = null;
let frameRoot = null;
let currentModel = null;
let textureLoader = null;
let spacePressed = false;
let perturbedFrameCache = new Map();
let perturbComputeSeq = 0;
let sceneLoadSeq = 0;

const form = document.getElementById("scene-form");
const taskConfig = document.getElementById("task-config");
const taskName = document.getElementById("task-name");
const seedInput = document.getElementById("seed");
const preGraspDisInput = document.getElementById("pre-grasp-dis");
const graspDisInput = document.getElementById("grasp-dis");
const refreshInput = document.getElementById("refresh");
const loadButton = document.getElementById("load-scene");
const showContactFrames = document.getElementById("show-contact-frames");
const showPreGraspFrames = document.getElementById("show-pre-grasp-frames");
const showTcpFrames = document.getElementById("show-tcp-frames");
const showGraspFrames = document.getElementById("show-grasp-frames");
const showPerturbedFrames = document.getElementById("show-perturbed-frames");
const showPerturbedContactFrames = document.getElementById("show-perturbed-contact-frames");
const showPerturbedTcpFrames = document.getElementById("show-perturbed-tcp-frames");
const showPerturbedPreGraspFrames = document.getElementById("show-perturbed-pre-grasp-frames");
const pointList = document.getElementById("point-list");
const drawButton = document.getElementById("draw-selected");
const clearButton = document.getElementById("clear-overlay");
const render3dButton = document.getElementById("render-3d");
const generatePreGraspButton = document.getElementById("generate-pre-grasp");
const resetPerturbButton = document.getElementById("reset-perturb");
const savePerturbButton = document.getElementById("save-perturb");
const perturbX = document.getElementById("perturb-x");
const perturbY = document.getElementById("perturb-y");
const perturbZ = document.getElementById("perturb-z");
const perturbR = document.getElementById("perturb-r");
const perturbP = document.getElementById("perturb-p");
const perturbYaw = document.getElementById("perturb-yaw");
const statusBox = document.getElementById("status");
const frame = document.getElementById("frame");
const view3d = document.getElementById("view3d");
const overlay = document.getElementById("overlay");
const metadata = document.getElementById("metadata");
const perturbPoseInfo = document.getElementById("perturb-pose-info");
const ctx = overlay.getContext("2d");
const overlayState = {
  hasDrawn: false,
  selectedPointKeys: [],
  showContactFrames: false,
  showPreGraspFrames: false,
  showTcpFrames: false,
  showGraspFrames: false,
  showPerturbedContactFrames: true,
  showPerturbedFrames: true,
  showPerturbedTcpFrames: true,
  showPerturbedPreGraspFrames: true,
};

async function init() {
  const res = await fetch("/api/configs");
  const data = await res.json();
  fillSelect(taskConfig, data.task_configs, "ep2_1_object_pose");
  fillSelect(taskName, data.tasks, data.default_task);
}

function fillSelect(select, values, preferred) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === preferred;
    select.appendChild(option);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await loadScene();
});

drawButton.addEventListener("click", drawSelectedPoints);
clearButton.addEventListener("click", clearOverlay);
render3dButton.addEventListener("click", reset3dView);
generatePreGraspButton.addEventListener("click", generatePreGraspPoses);
resetPerturbButton.addEventListener("click", resetPerturbation);
savePerturbButton.addEventListener("click", savePerturbation);
preGraspDisInput.addEventListener("input", handlePerturbationChanged);
graspDisInput.addEventListener("input", handlePerturbationChanged);
showContactFrames.addEventListener("change", () => {
  overlayState.showContactFrames = showContactFrames.checked;
  refreshVisualizations();
});
showPreGraspFrames.addEventListener("change", () => {
  overlayState.showPreGraspFrames = showPreGraspFrames.checked;
  refreshVisualizations();
});
showTcpFrames.addEventListener("change", () => {
  overlayState.showTcpFrames = showTcpFrames.checked;
  refreshVisualizations();
});
showGraspFrames.addEventListener("change", () => {
  overlayState.showGraspFrames = showGraspFrames.checked;
  refreshVisualizations();
});
showPerturbedContactFrames.addEventListener("change", () => {
  overlayState.showPerturbedContactFrames = showPerturbedContactFrames.checked;
  refreshVisualizations();
});
showPerturbedFrames.addEventListener("change", () => {
  overlayState.showPerturbedFrames = showPerturbedFrames.checked;
  refreshVisualizations();
});
showPerturbedTcpFrames.addEventListener("change", () => {
  overlayState.showPerturbedTcpFrames = showPerturbedTcpFrames.checked;
  refreshVisualizations();
});
showPerturbedPreGraspFrames.addEventListener("change", () => {
  overlayState.showPerturbedPreGraspFrames = showPerturbedPreGraspFrames.checked;
  refreshVisualizations();
});
window.addEventListener("keydown", handlePerturbKeydown);
window.addEventListener("keyup", handlePerturbKeyup);
window.addEventListener("resize", refreshVisualizations);
frame.addEventListener("load", refreshVisualizations);
if (window.ResizeObserver) {
  new ResizeObserver(renderOverlayFromState).observe(document.querySelector(".image-wrap"));
  new ResizeObserver(render3dScene).observe(document.querySelector(".open3d-wrap"));
}
setup3dInteraction();
[perturbX, perturbY, perturbZ, perturbR, perturbP, perturbYaw].forEach((input) => {
  input.addEventListener("input", handlePerturbationChanged);
});

async function loadScene() {
  const seq = ++sceneLoadSeq;
  perturbComputeSeq += 1;
  const forceRefresh = refreshInput.checked;
  const requestedScene = currentSceneRequest();
  setStatus(forceRefresh ? "正在重新生成缓存..." : "正在生成 / 加载首帧...");
  loadButton.disabled = true;
  clearOverlay();
  const params = new URLSearchParams({
    task_config: requestedScene.taskConfig,
    task_name: requestedScene.taskName,
    seed: String(requestedScene.seed),
    refresh: forceRefresh ? "1" : "0",
  });

  try {
    const res = await fetch(`/api/scene?${params.toString()}`);
    const data = await res.json();
    if (seq !== sceneLoadSeq) return;
    if (!res.ok) {
      setStatus(`加载失败：${data.error || JSON.stringify(data)}`);
      return;
    }
    if (!matchesSceneRequest(requestedScene) || Number(data.seed) !== requestedScene.seed) {
      return;
    }

    sceneData = data;
    objectAsset = null;
    currentModel = null;
    if (objectRoot) objectRoot.clear();
    if (frameRoot) frameRoot.clear();
    perturbedFrameCache = new Map();
    resetPerturbationInputs();
    overlayState.hasDrawn = false;
    overlayState.selectedPointKeys = [];
    await loadFrameImage(`${sceneData.image_url}?t=${Date.now()}`, seq);
    if (seq !== sceneLoadSeq) return;
    syncCanvasSize();
    createEnvironment3d();
    renderPointList();
    metadata.textContent = JSON.stringify(buildMetadata(sceneData), null, 2);
    drawButton.disabled = false;
    clearButton.disabled = false;
    render3dButton.disabled = false;
    generatePreGraspButton.disabled = false;
    resetPerturbButton.disabled = false;
    savePerturbButton.disabled = false;
    await loadObjectMesh(seq);
    if (seq !== sceneLoadSeq) return;

    const cacheText = sceneData.cache_hit ? "已复用缓存。" : "已重新生成缓存。";
    const graspText = hasRobotwinGraspMatrices() ? "" : " 当前缓存缺少 RoboTwin grasp 映射，请勾选“重新生成缓存”刷新。";
    setStatus(`${cacheText}${graspText} 初始不显示坐标系，点击“可视化选中点”后绘制。右侧 3D 左键旋转，右键平移中心，滚轮缩放。`);
  } finally {
    if (seq === sceneLoadSeq) {
      loadButton.disabled = false;
    }
  }
}

function loadFrameImage(src, seq) {
  frame.removeAttribute("src");
  syncCanvasSize();
  clearCanvasOnly();
  return new Promise((resolve, reject) => {
    frame.onload = () => {
      if (seq !== sceneLoadSeq) {
        resolve();
        return;
      }
      if (frame.decode) {
        frame.decode().then(resolve).catch(resolve);
      } else {
        resolve();
      }
    };
    frame.onerror = () => {
      if (seq === sceneLoadSeq) reject(new Error("首帧图片加载失败。"));
      else resolve();
    };
    frame.src = src;
  });
}

function hasRobotwinGraspMatrices() {
  return getAllPoints().every((point) => point.grasp_matrix_world);
}

function renderPointList() {
  const points = getAllPoints();
  pointList.innerHTML = "";
  pointList.classList.toggle("empty", points.length === 0);
  if (!points.length) {
    pointList.textContent = "没有 contact point";
    return;
  }

  points.forEach((point) => {
    const label = document.createElement("label");
    label.className = "check-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = point.key;
    input.checked = false;
    input.addEventListener("change", handlePerturbationChanged);
    label.appendChild(input);
    label.appendChild(document.createTextNode(`${point.objectName} / point ${point.id}`));
    pointList.appendChild(label);
  });
}

function getAllPoints() {
  if (!sceneData) return [];
  return sceneData.objects.flatMap((object) =>
    object.contact_points.map((point) => ({
      ...point,
      objectName: object.name,
      key: `${object.name}:${point.id}`,
    })),
  );
}

function drawSelectedPoints() {
  overlayState.hasDrawn = true;
  overlayState.selectedPointKeys = getSelectedPointKeys();
  overlayState.showContactFrames = showContactFrames.checked;
  overlayState.showPreGraspFrames = showPreGraspFrames.checked;
  overlayState.showTcpFrames = showTcpFrames.checked;
  overlayState.showGraspFrames = showGraspFrames.checked;
  overlayState.showPerturbedContactFrames = showPerturbedContactFrames.checked;
  overlayState.showPerturbedFrames = showPerturbedFrames.checked;
  overlayState.showPerturbedTcpFrames = showPerturbedTcpFrames.checked;
  overlayState.showPerturbedPreGraspFrames = showPerturbedPreGraspFrames.checked;
  renderOverlayFromState();
  render3dScene();
  updatePerturbPoseInfo();
}

async function loadObjectMesh(seq = sceneLoadSeq) {
  if (!sceneData) return;
  render3dButton.disabled = true;
  const params = new URLSearchParams({
    task_config: taskConfig.value,
    task_name: taskName.value,
    seed: seedInput.value || "0",
  });

  try {
    const res = await fetch(`/api/object_asset?${params.toString()}`);
    const data = await res.json();
    if (!res.ok) {
      if (seq !== sceneLoadSeq) return;
      setStatus(`3D 模型加载失败：${data.error || JSON.stringify(data)}`);
      return;
    }
    if (seq !== sceneLoadSeq) return;
    objectAsset = data;
    await loadTexturedModel(data, seq);
  } catch (error) {
    if (seq === sceneLoadSeq) {
      setStatus(`3D 模型加载失败：${error.message}`);
    }
  } finally {
    if (seq === sceneLoadSeq) {
      render3dButton.disabled = false;
    }
  }
}

function getSelectedPointIds() {
  return getSelectedPointKeys().map((key) => {
    const keyParts = key.split(":");
    return Number.parseInt(keyParts[keyParts.length - 1], 10);
  });
}

function getSelectedPointKeys() {
  return [...pointList.querySelectorAll("input[type='checkbox']:checked")].map((input) => input.value);
}

function getPerturbation() {
  return {
    x: readNumber(perturbX),
    y: readNumber(perturbY),
    z: readNumber(perturbZ),
    r: readNumber(perturbR),
    p: readNumber(perturbP),
    yaw: readNumber(perturbYaw),
  };
}

function currentSceneRequest() {
  return {
    taskConfig: taskConfig.value,
    taskName: taskName.value,
    seed: Number.parseInt(seedInput.value || "0", 10),
  };
}

function currentSceneIdentity() {
  if (!sceneData) return null;
  return {
    taskConfig: sceneData.task_config,
    taskName: sceneData.task_name,
    seed: Number(sceneData.seed),
  };
}

function matchesSceneRequest(request) {
  const current = currentSceneRequest();
  return (
    current.taskConfig === request.taskConfig
    && current.taskName === request.taskName
    && current.seed === request.seed
  );
}

function matchesLoadedScene(identity) {
  const current = currentSceneIdentity();
  return Boolean(current)
    && current.taskConfig === identity.taskConfig
    && current.taskName === identity.taskName
    && current.seed === identity.seed
    && matchesSceneRequest(identity);
}

function perturbationSignature(perturbation = getPerturbation()) {
  return ["x", "y", "z", "r", "p", "yaw"]
    .map((key) => `${key}:${Number(perturbation[key] || 0).toPrecision(12)}`)
    .join("|");
}

function perturbedFrameCacheKey(
  pointId,
  identity = currentSceneIdentity(),
  perturbation = getPerturbation(),
  preGraspDis = getPreGraspDis(),
  graspDis = getGraspDis(),
) {
  if (!identity) return null;
  return [
    identity.taskConfig,
    identity.taskName,
    identity.seed,
    Number(pointId),
    Number(preGraspDis).toPrecision(12),
    Number(graspDis).toPrecision(12),
    perturbationSignature(perturbation),
  ].join("::");
}

function resetPerturbation() {
  resetPerturbationInputs();
  handlePerturbationChanged();
}

function resetPerturbationInputs() {
  [perturbX, perturbY, perturbZ, perturbR, perturbP, perturbYaw].forEach((input) => {
    input.value = "0";
  });
}

function handlePerturbKeydown(event) {
  if (!sceneData || isTypingTarget(event.target)) return;
  if (event.code === "Space") {
    spacePressed = true;
    event.preventDefault();
    return;
  }
  const direction = spacePressed ? -1 : 1;
  const positionStep = 0.005 * direction;
  const rotationStep = 0.03 * direction;
  const actions = {
    ArrowUp: () => addInputValue(perturbX, positionStep),
    ArrowLeft: () => addInputValue(perturbY, positionStep),
    ArrowRight: () => addInputValue(perturbZ, positionStep),
    w: () => addInputValue(perturbR, rotationStep),
    a: () => addInputValue(perturbP, rotationStep),
    d: () => addInputValue(perturbYaw, rotationStep),
  };
  const action = actions[event.key] || actions[event.key.toLowerCase()];
  if (!action) return;
  event.preventDefault();
  action();
  handlePerturbationChanged();
}

function handlePerturbKeyup(event) {
  if (event.code === "Space") {
    spacePressed = false;
    event.preventDefault();
  }
}

function isTypingTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName) || target.isContentEditable;
}

function addInputValue(input, delta) {
  input.value = (readNumber(input) + delta).toFixed(delta < 0.01 && delta > -0.01 ? 3 : 4);
}

async function savePerturbation() {
  if (!sceneData) return;
  const selected = new Set(getSelectedPointKeys());
  const points = getAllPoints().filter((point) => selected.has(point.key));
  if (!points.length) {
    setStatus("请先选择 contact point。");
    return;
  }

  if (!validateGraspDistances()) return;
  const preGraspDis = getPreGraspDis();
  const graspDis = getGraspDis();
  const items = points.map((point) => {
    const frames = computeGraspFrameMatrices(point, preGraspDis, graspDis);
    if (!frames.perturbedTcp || !frames.perturbedGrasp || !frames.perturbedPreGrasp) return null;
    return {
      point_id: point.id,
      perturbed_contact_matrix_world: frames.perturbedContact,
      perturbed_tcp_matrix_world: frames.perturbedTcp,
      perturbed_grasp_matrix_world: frames.perturbedGrasp,
      perturbed_pre_grasp_matrix_world: frames.perturbedPreGrasp,
      perturbed_contact_pose_world: matrixToPose(frames.perturbedContact),
      perturbed_tcp_pose_world: matrixToPose(frames.perturbedTcp),
      perturbed_grasp_pose_world: matrixToPose(frames.perturbedGrasp),
      perturbed_pre_grasp_pose_world: matrixToPose(frames.perturbedPreGrasp),
    };
  }).filter(Boolean);
  if (!items.length) {
    setStatus("没有可保存的 RoboTwin 抓取位姿。请先点击“生成预抓取位姿”。");
    return;
  }

  savePerturbButton.disabled = true;
  try {
    drawSelectedPoints();
    const imageDataUrl = captureFrameWithAxes();
    const res = await fetch("/api/save_perturbation", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        task_config: taskConfig.value,
        task_name: taskName.value,
        seed: Number.parseInt(seedInput.value || "0", 10),
        pre_grasp_dis: preGraspDis,
        grasp_dis: graspDis,
        selected_point_ids: getSelectedPointIds(),
        perturbation: getPerturbation(),
        poses: items,
        image_data_url: imageDataUrl,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(`保存失败：${data.error || JSON.stringify(data)}`);
      return;
    }
    setStatus(`已保存 ${data.saved_count} 个扰动位姿和坐标系图片：${data.path}`);
  } catch (error) {
    setStatus(`保存失败：${error.message}`);
  } finally {
    savePerturbButton.disabled = false;
  }
}

function captureFrameWithAxes() {
  if (!frame.complete || !frame.naturalWidth || !frame.naturalHeight) {
    throw new Error("首帧图片还没有加载完成。");
  }
  const canvas = document.createElement("canvas");
  canvas.width = frame.naturalWidth;
  canvas.height = frame.naturalHeight;
  const captureCtx = canvas.getContext("2d");
  captureCtx.drawImage(frame, 0, 0, canvas.width, canvas.height);
  captureCtx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

function readNumber(input) {
  const value = Number.parseFloat(input.value);
  return Number.isFinite(value) ? value : 0;
}

function refreshVisualizations() {
  if (overlayState.hasDrawn) {
    overlayState.selectedPointKeys = getSelectedPointKeys();
  }
  updatePerturbPoseInfo();
  renderOverlayFromState();
  render3dScene();
}

function handlePerturbationChanged() {
  perturbedFrameCache = new Map();
  perturbComputeSeq += 1;
  if (sceneData && overlayState.hasDrawn) {
    overlayState.selectedPointKeys = getSelectedPointKeys();
  }
  updatePerturbPoseInfo();
  renderOverlayFromState();
  render3dScene();
  if (sceneData && getSelectedPointKeys().length) {
    setStatus("扰动 contact point 坐标系已更新。点击“生成预抓取位姿”后进行 RoboTwin 规划。");
  }
}

async function generatePreGraspPoses() {
  return computePerturbedGraspsFromRobotwin();
}

async function computePerturbedGraspsFromRobotwin() {
  const selectedIds = getSelectedPointIds();
  if (!selectedIds.length) {
    setStatus("请先选择 contact point。");
    return false;
  }
  if (!validateGraspDistances()) return false;
  const seq = ++perturbComputeSeq;
  const requestScene = currentSceneIdentity();
  if (!requestScene) return false;
  const requestPerturbation = getPerturbation();
  const requestPreGraspDis = getPreGraspDis();
  const requestGraspDis = getGraspDis();
  generatePreGraspButton.disabled = true;
  setStatus("正在基于扰动 contact point 规划预抓取 / 抓取 / TCP 位姿...");
  try {
    const res = await fetch("/api/compute_perturbed_grasps", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        task_config: requestScene.taskConfig,
        task_name: requestScene.taskName,
        seed: requestScene.seed,
        selected_point_ids: selectedIds,
        pre_grasp_dis: requestPreGraspDis,
        grasp_dis: requestGraspDis,
        perturbation: requestPerturbation,
      }),
    });
    const data = await res.json();
    if (seq !== perturbComputeSeq || !matchesLoadedScene(requestScene)) return false;
    if (!res.ok) {
      setStatus(`RoboTwin 抓取位姿计算失败：${data.error || JSON.stringify(data)}`);
      return false;
    }
    perturbedFrameCache = new Map((data.items || []).map((item) => [
      perturbedFrameCacheKey(
        item.point_id,
        requestScene,
        requestPerturbation,
        requestPreGraspDis,
        requestGraspDis,
      ),
      item,
    ]));
    updatePerturbPoseInfo();
    renderOverlayFromState();
    render3dScene();
    const hasGrasp = (data.items || []).some((item) => item.perturbed_grasp_matrix_world);
    setStatus(hasGrasp ? "预抓取 / 抓取 / TCP 位姿已生成。" : "RoboTwin 没有返回可用扰动抓取位姿。");
    return true;
  } catch (error) {
    if (seq === perturbComputeSeq) {
      setStatus(`RoboTwin 抓取位姿计算失败：${error.message}`);
    }
    return false;
  } finally {
    generatePreGraspButton.disabled = !sceneData;
  }
}

function updatePerturbPoseInfo() {
  if (!sceneData) {
    perturbPoseInfo.textContent = "选择 contact point 后显示扰动位姿。";
    return;
  }
  const selected = new Set(getSelectedPointKeys());
  const points = getAllPoints().filter((point) => selected.has(point.key));
  if (!points.length) {
    perturbPoseInfo.textContent = "选择 contact point 后显示扰动位姿。";
    return;
  }

  const blocks = [];
  points.forEach((point) => {
    const frames = computeGraspFrameMatrices(point, getPreGraspDis(), getGraspDis());
    blocks.push(`
      <section class="pose-point">
        <div class="pose-point-title">point ${point.id}</div>
        ${formatPoseBlock("Perturbed Contact", frames.perturbedContact)}
        ${formatPoseBlock("Perturbed TCP", frames.perturbedTcp)}
        ${formatPoseBlock("Perturbed Grasp", frames.perturbedGrasp)}
        ${formatPoseBlock("Perturbed Pre-Grasp", frames.perturbedPreGrasp)}
      </section>
    `);
  });
  perturbPoseInfo.innerHTML = blocks.join("");
}

function formatPoseBlock(name, matrix) {
  if (!matrix) {
    return `
      <div class="pose-block">
        <div class="pose-name">${name}</div>
        <div class="pose-row"><span>等待 RoboTwin 计算</span></div>
      </div>
    `;
  }
  const xyz = [matrix[0][3], matrix[1][3], matrix[2][3]];
  const rpy = eulerXyzFromMatrix(matrix);
  const eulerPose = [...xyz, ...rpy].map(formatNumber).join(", ");
  const quaternionPose = matrixToPose(matrix).map(formatNumber).join(", ");
  return `
    <div class="pose-block">
      <div class="pose-name">${name}</div>
      <div class="pose-row"><span>Euler angles</span><code>[${eulerPose}]</code></div>
      <div class="pose-row"><span>Quaternion</span><code>[${quaternionPose}]</code></div>
    </div>
  `;
}

function formatNumber(value) {
  return Number(value).toFixed(4);
}

function renderOverlayFromState() {
  syncCanvasSize();
  clearCanvasOnly();
  if (!sceneData || !overlayState.hasDrawn) return;

  const selected = new Set(overlayState.selectedPointKeys);
  getAllPoints().forEach((point) => {
    if (selected.has(point.key)) {
      drawPointAxes(point);
    }
  });
}

function drawPointAxes(point) {
  const frames = computeGraspFrames(point, getPreGraspDis(), getGraspDis());
  if (overlayState.showContactFrames) {
    drawFrame(point.axes_2d, point.projection_valid, `cp${point.id}`, false);
  }
  if (overlayState.showPreGraspFrames) {
    drawFrame(frames.preGrasp.axes_2d, frames.preGrasp.valid, `pre${point.id}`, true);
  }
  if (overlayState.showTcpFrames) {
    drawFrame(frames.tcp.axes_2d, frames.tcp.valid, `tcp${point.id}`, [7, 3]);
  }
  if (overlayState.showGraspFrames) {
    drawFrame(frames.grasp.axes_2d, frames.grasp.valid, `grasp${point.id}`, false);
  }
  if (overlayState.showPerturbedContactFrames) {
    drawFrame(frames.perturbedContact.axes_2d, frames.perturbedContact.valid, `pcp${point.id}`, [2, 3]);
  }
  if (overlayState.showPerturbedFrames) {
    drawFrame(frames.perturbedGrasp.axes_2d, frames.perturbedGrasp.valid, `per${point.id}`, [2, 3, 8, 3]);
  }
  if (overlayState.showPerturbedTcpFrames) {
    drawFrame(frames.perturbedTcp.axes_2d, frames.perturbedTcp.valid, `ptcp${point.id}`, [2, 3]);
  }
  if (overlayState.showPerturbedPreGraspFrames) {
    drawFrame(frames.perturbedPreGrasp.axes_2d, frames.perturbedPreGrasp.valid, `ppre${point.id}`, [8, 3]);
  }
}

function drawFrame(axes, valid, label, dashed) {
  if (!valid || !axes || !axes.origin) return;
  drawAxis(axes.origin, axes.x, "#e53935", "x", dashed);
  drawAxis(axes.origin, axes.y, "#2e7d32", "y", dashed);
  drawAxis(axes.origin, axes.z, "#1565c0", "z", dashed);
  const [ox, oy] = scalePoint(axes.origin);
  ctx.beginPath();
  ctx.arc(ox, oy, 4, 0, Math.PI * 2);
  ctx.fillStyle = "#111111";
  ctx.fill();
  ctx.font = "13px Arial";
  ctx.fillText(label, ox + 6, oy - 6);
}

function drawAxis(origin, endpoint, color, label, dashed) {
  if (!origin || !endpoint) return;
  const [x1, y1] = scalePoint(origin);
  const [x2, y2] = scalePoint(endpoint);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = dashed ? 1.5 : 2;
  ctx.setLineDash(Array.isArray(dashed) ? dashed : (dashed ? [5, 4] : []));
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  drawArrowHead(x1, y1, x2, y2);
  ctx.font = "12px Arial";
  ctx.fillText(label, x2 + 3, y2 + 3);
}

function drawArrowHead(x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const size = 7;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - size * Math.cos(angle - Math.PI / 6), y2 - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - size * Math.cos(angle + Math.PI / 6), y2 - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

function scalePoint(point) {
  const rect = frame.getBoundingClientRect();
  const sx = rect.width / sceneData.image_width;
  const sy = rect.height / sceneData.image_height;
  return [point[0] * sx, point[1] * sy];
}

function syncCanvasSize() {
  const rect = frame.getBoundingClientRect();
  const wrapRect = frame.parentElement.getBoundingClientRect();
  overlay.width = Math.round(rect.width);
  overlay.height = Math.round(rect.height);
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  overlay.style.left = `${rect.left - wrapRect.left}px`;
  overlay.style.top = `${rect.top - wrapRect.top}px`;
}

function clearOverlay() {
  overlayState.hasDrawn = false;
  overlayState.selectedPointKeys = [];
  perturbedFrameCache = new Map();
  updatePerturbPoseInfo();
  syncCanvasSize();
  clearCanvasOnly();
}

function clearCanvasOnly() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function getPreGraspDis() {
  const value = Number.parseFloat(preGraspDisInput.value);
  return Number.isFinite(value) && value >= 0 ? value : 0.1;
}

function getGraspDis() {
  const value = Number.parseFloat(graspDisInput.value);
  return Number.isFinite(value) && value >= 0 ? value : 0.0;
}

function validateGraspDistances() {
  if (getPreGraspDis() < getGraspDis()) {
    setStatus("pre_grasp_dis 必须大于或等于 grasp_dis。");
    return false;
  }
  return true;
}

function computeGraspFrames(point, preGraspDis, graspDis) {
  const matrices = computeGraspFrameMatrices(point, preGraspDis, graspDis);
  return {
    preGrasp: projectFrame(matrices.preGrasp),
    tcp: projectFrame(matrices.tcp),
    grasp: projectFrame(matrices.grasp),
    perturbedContact: projectFrame(matrices.perturbedContact),
    perturbedTcp: projectFrame(matrices.perturbedTcp),
    perturbedGrasp: projectFrame(matrices.perturbedGrasp),
    perturbedPreGrasp: projectFrame(matrices.perturbedPreGrasp),
  };
}

function computeGraspFrameMatrices(point, preGraspDis = getPreGraspDis(), graspDis = getGraspDis()) {
  const contactMatrix = point.matrix_world;
  const contactToGrasp = contactToGraspMatrix(point);
  const contactGraspMatrix = multiplyMat4(contactMatrix, contactToGrasp);
  const graspMatrix = translatedAlongLocalX(contactGraspMatrix, -graspDis);
  const tcpMatrix = translatedAlongLocalX(graspMatrix, 0.12);
  const preGraspMatrix = translatedAlongLocalX(graspMatrix, -(preGraspDis - graspDis));
  const perturbedContactMatrix = multiplyMat4(contactMatrix, deltaMatrix(getPerturbation()));
  const perturbedContactGraspMatrix = multiplyMat4(perturbedContactMatrix, contactToGrasp);
  const perturbedGraspMatrix = translatedAlongLocalX(perturbedContactGraspMatrix, -graspDis);
  const perturbedTcpMatrix = translatedAlongLocalX(perturbedGraspMatrix, 0.12);
  const perturbedPreGraspMatrix = translatedAlongLocalX(perturbedGraspMatrix, -(preGraspDis - graspDis));
  const robotwinFrames = perturbedFrameCache.get(perturbedFrameCacheKey(point.id));
  return {
    preGrasp: preGraspMatrix,
    tcp: tcpMatrix,
    grasp: graspMatrix,
    perturbedContact: robotwinFrames?.perturbed_contact_matrix_world || perturbedContactMatrix,
    perturbedTcp: robotwinFrames?.perturbed_tcp_matrix_world || null,
    perturbedGrasp: robotwinFrames?.perturbed_grasp_matrix_world || null,
    perturbedPreGrasp: robotwinFrames?.perturbed_pre_grasp_matrix_world || null,
  };
}

function contactToGraspMatrix(point) {
  if (point.grasp_matrix_world) {
    return multiplyMat4(invertRigidMat4(point.matrix_world), point.grasp_matrix_world);
  }
  const contactToTcp = [
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
  ];
  return translatedAlongLocalX(contactToTcp, -0.12);
}

function invertRigidMat4(matrix) {
  const result = [
    [matrix[0][0], matrix[1][0], matrix[2][0], 0],
    [matrix[0][1], matrix[1][1], matrix[2][1], 0],
    [matrix[0][2], matrix[1][2], matrix[2][2], 0],
    [0, 0, 0, 1],
  ];
  const translation = [matrix[0][3], matrix[1][3], matrix[2][3]];
  result[0][3] = -(result[0][0] * translation[0] + result[0][1] * translation[1] + result[0][2] * translation[2]);
  result[1][3] = -(result[1][0] * translation[0] + result[1][1] * translation[1] + result[1][2] * translation[2]);
  result[2][3] = -(result[2][0] * translation[0] + result[2][1] * translation[1] + result[2][2] * translation[2]);
  return result;
}

function translatedAlongLocalX(matrix, distance) {
  const result = cloneMat4(matrix);
  result[0][3] += matrix[0][0] * distance;
  result[1][3] += matrix[1][0] * distance;
  result[2][3] += matrix[2][0] * distance;
  return result;
}

function projectFrame(matrixWorld) {
  if (!matrixWorld) return {axes_2d: null, valid: false};
  const axisLength = sceneData.axis_length || 0.04;
  const origin = [matrixWorld[0][3], matrixWorld[1][3], matrixWorld[2][3]];
  const axes3d = {
    origin,
    x: [
      origin[0] + matrixWorld[0][0] * axisLength,
      origin[1] + matrixWorld[1][0] * axisLength,
      origin[2] + matrixWorld[2][0] * axisLength,
    ],
    y: [
      origin[0] + matrixWorld[0][1] * axisLength,
      origin[1] + matrixWorld[1][1] * axisLength,
      origin[2] + matrixWorld[2][1] * axisLength,
    ],
    z: [
      origin[0] + matrixWorld[0][2] * axisLength,
      origin[1] + matrixWorld[1][2] * axisLength,
      origin[2] + matrixWorld[2][2] * axisLength,
    ],
  };
  const axes2d = Object.fromEntries(
    Object.entries(axes3d).map(([name, point]) => [name, projectPoint(point)]),
  );
  return {
    axes_2d: axes2d,
    valid: Object.values(axes2d).every((point) => point !== null),
  };
}

function projectPoint(pointWorld) {
  const pointCam = transformPoint(sceneData.camera.extrinsic, pointWorld);
  if (pointCam[2] <= 1e-6) return null;
  const uvw = multiplyMat3Vec3(sceneData.camera.intrinsic, pointCam);
  return [uvw[0] / uvw[2], uvw[1] / uvw[2]];
}

function transformPoint(matrix, point) {
  return [
    matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
    matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
    matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3],
  ];
}

function multiplyMat3Vec3(matrix, vector) {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
  ];
}

function multiplyMat4(a, b) {
  return a.map((row, i) => b[0].map((_, j) =>
    row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j] + row[3] * b[3][j],
  ));
}

function cloneMat4(matrix) {
  return matrix.map((row) => row.slice());
}

function matrixToPose(matrix) {
  const q = quaternionFromMatrix(matrix);
  return [matrix[0][3], matrix[1][3], matrix[2][3], q[0], q[1], q[2], q[3]];
}

function eulerXyzFromMatrix(matrix) {
  const pitch = Math.asin(Math.max(-1, Math.min(1, -matrix[2][0])));
  const cosPitch = Math.cos(pitch);
  if (Math.abs(cosPitch) > 1e-6) {
    return [
      Math.atan2(matrix[2][1], matrix[2][2]),
      pitch,
      Math.atan2(matrix[1][0], matrix[0][0]),
    ];
  }
  return [
    Math.atan2(-matrix[1][2], matrix[1][1]),
    pitch,
    0,
  ];
}

function quaternionFromMatrix(matrix) {
  const m00 = matrix[0][0], m01 = matrix[0][1], m02 = matrix[0][2];
  const m10 = matrix[1][0], m11 = matrix[1][1], m12 = matrix[1][2];
  const m20 = matrix[2][0], m21 = matrix[2][1], m22 = matrix[2][2];
  const trace = m00 + m11 + m22;
  let qw, qx, qy, qz;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1.0) * 2;
    qw = 0.25 * s;
    qx = (m21 - m12) / s;
    qy = (m02 - m20) / s;
    qz = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2;
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2;
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2;
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }
  const norm = Math.hypot(qw, qx, qy, qz) || 1;
  return [qw / norm, qx / norm, qy / norm, qz / norm];
}

function deltaMatrix(values) {
  const [sr, cr] = [Math.sin(values.r), Math.cos(values.r)];
  const [sp, cp] = [Math.sin(values.p), Math.cos(values.p)];
  const [sy, cy] = [Math.sin(values.yaw), Math.cos(values.yaw)];
  const rotation = [
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ];
  return [
    [rotation[0][0], rotation[0][1], rotation[0][2], values.x],
    [rotation[1][0], rotation[1][1], rotation[1][2], values.y],
    [rotation[2][0], rotation[2][1], rotation[2][2], values.z],
    [0, 0, 0, 1],
  ];
}

function setup3dInteraction() {
  threeRenderer = new THREE.WebGLRenderer({canvas: view3d, antialias: true});
  threeRenderer.setClearColor(0xffffff, 1);
  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  threeScene = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 10);
  threeCamera.up.set(0, 0, 1);
  threeCamera.position.set(0.45, -0.65, 1.1);

  threeControls = new OrbitControls(threeCamera, view3d);
  threeControls.enableDamping = true;
  threeControls.enablePan = true;
  threeControls.screenSpacePanning = true;
  threeControls.target.set(0, 0, 0.8);

  threeScene.add(new THREE.AmbientLight(0xffffff, 1.1));
  const light = new THREE.DirectionalLight(0xffffff, 1.8);
  light.position.set(1, -1.5, 2);
  threeScene.add(light);

  objectRoot = new THREE.Group();
  environmentRoot = new THREE.Group();
  frameRoot = new THREE.Group();
  threeScene.add(environmentRoot);
  threeScene.add(objectRoot);
  threeScene.add(frameRoot);
  gltfLoader = new GLTFLoader();
  textureLoader = new THREE.TextureLoader();
  threeRenderer.setAnimationLoop(() => {
    sync3dCanvasSize();
    threeControls.update();
    threeRenderer.render(threeScene, threeCamera);
  });
}

function render3dScene() {
  if (!threeRenderer) return;
  sync3dCanvasSize();
  updateFrameObjects3d();
}

function sync3dCanvasSize() {
  if (!threeRenderer) return;
  const frameRect = frame.getBoundingClientRect();
  const wrapRect = view3d.parentElement.getBoundingClientRect();
  const width = Math.max(1, Math.round(Math.min(frameRect.width, wrapRect.width)));
  const height = Math.max(1, Math.round(Math.min(frameRect.height, wrapRect.height)));
  view3d.style.width = `${width}px`;
  view3d.style.height = `${height}px`;
  if (view3d.width !== width || view3d.height !== height) {
    threeRenderer.setSize(width, height, false);
    threeCamera.aspect = width / height;
    threeCamera.updateProjectionMatrix();
  }
}

async function loadTexturedModel(asset, seq = sceneLoadSeq) {
  objectRoot.clear();
  currentModel = null;
  await new Promise((resolve, reject) => {
    gltfLoader.load(asset.mesh_url, (gltf) => {
      if (seq !== sceneLoadSeq) {
        resolve();
        return;
      }
      const model = gltf.scene;
      model.traverse((node) => {
        if (node.isMesh && node.material) {
          node.material.side = THREE.DoubleSide;
          node.material.needsUpdate = true;
        }
      });
      model.matrixAutoUpdate = false;
      model.matrix.copy(objectWorldMatrix(asset));
      objectRoot.add(model);
      currentModel = model;
      reset3dView();
      updateFrameObjects3d();
      resolve();
    }, undefined, reject);
  });
}

function updateFrameObjects3d() {
  if (!frameRoot || !sceneData) return;
  frameRoot.clear();
  const selected = new Set(overlayState.selectedPointKeys);
  getAllPoints().forEach((point) => {
    if (!selected.has(point.key)) return;
    const frames = computeGraspFrameMatrices(point, getPreGraspDis(), getGraspDis());
    if (overlayState.showContactFrames) frameRoot.add(makeFrame3d(point.matrix_world, 0.04, `cp${point.id}`));
    if (overlayState.showPreGraspFrames) frameRoot.add(makeFrame3d(frames.preGrasp, 0.05, `pre${point.id}`));
    if (overlayState.showTcpFrames) frameRoot.add(makeFrame3d(frames.tcp, 0.045, `tcp${point.id}`));
    if (overlayState.showGraspFrames) frameRoot.add(makeFrame3d(frames.grasp, 0.05, `grasp${point.id}`));
    if (overlayState.showPerturbedContactFrames) frameRoot.add(makeFrame3d(frames.perturbedContact, 0.045, `pcp${point.id}`));
    if (overlayState.showPerturbedTcpFrames) frameRoot.add(makeFrame3d(frames.perturbedTcp, 0.052, `ptcp${point.id}`));
    if (overlayState.showPerturbedFrames) frameRoot.add(makeFrame3d(frames.perturbedGrasp, 0.055, `pgrasp${point.id}`));
    if (overlayState.showPerturbedPreGraspFrames) frameRoot.add(makeFrame3d(frames.perturbedPreGrasp, 0.055, `ppre${point.id}`));
  });
}

function makeFrame3d(matrixRows, length, label) {
  const group = new THREE.Group();
  if (!matrixRows) return group;
  const origin = new THREE.Vector3(matrixRows[0][3], matrixRows[1][3], matrixRows[2][3]);
  const axes = [
    {color: 0xe53935, dir: new THREE.Vector3(matrixRows[0][0], matrixRows[1][0], matrixRows[2][0])},
    {color: 0x2e7d32, dir: new THREE.Vector3(matrixRows[0][1], matrixRows[1][1], matrixRows[2][1])},
    {color: 0x1565c0, dir: new THREE.Vector3(matrixRows[0][2], matrixRows[1][2], matrixRows[2][2])},
  ];
  axes.forEach(({color, dir}) => {
    group.add(makeAxisMesh(origin, dir.normalize(), length, color));
  });
  group.add(makeTextSprite(label, origin.clone().add(new THREE.Vector3(length * 0.12, length * 0.12, length * 0.12))));
  return group;
}

function makeTextSprite(text, position) {
  const canvas = document.createElement("canvas");
  canvas.width = 192;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.font = "24px Arial";
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#222222";
  context.lineWidth = 2;
  context.strokeText(text, 12, 40);
  context.fillStyle = "#111111";
  context.fillText(text, 12, 40);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, transparent: true, depthTest: false}));
  sprite.position.copy(position);
  sprite.scale.set(0.042, 0.014, 1);
  return sprite;
}

function makeAxisMesh(origin, direction, length, color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({color, roughness: 0.45});
  const shaftLength = length * 0.78;
  const radius = length * 0.028;
  const midpoint = origin.clone().add(direction.clone().multiplyScalar(shaftLength / 2));
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, shaftLength, 12), material);
  shaft.position.copy(midpoint);
  shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  group.add(shaft);

  const head = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.3, length * 0.20, 16), material);
  head.position.copy(origin.clone().add(direction.clone().multiplyScalar(length * 0.89)));
  head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  group.add(head);
  return group;
}

function objectWorldMatrix(asset) {
  const pose = new THREE.Matrix4().compose(
    new THREE.Vector3(...asset.pose_world.p),
    new THREE.Quaternion(asset.pose_world.q[1], asset.pose_world.q[2], asset.pose_world.q[3], asset.pose_world.q[0]),
    new THREE.Vector3(1, 1, 1),
  );
  const scale = new THREE.Matrix4().makeScale(...asset.scale);
  return pose.multiply(scale);
}

function matrix4FromRows(rows) {
  return new THREE.Matrix4().set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3],
    rows[3][0], rows[3][1], rows[3][2], rows[3][3],
  );
}

function frameCameraOnObject(model) {
  const box = new THREE.Box3().setFromObject(environmentRoot);
  box.union(new THREE.Box3().setFromObject(model));
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  const radius = Math.max(size * 0.28, 0.45);
  threeControls.target.copy(center);
  threeCamera.position.copy(center).add(new THREE.Vector3(radius * 1.2, -radius * 1.8, radius * 1.2));
  threeCamera.near = 0.01;
  threeCamera.far = 20;
  threeCamera.updateProjectionMatrix();
  threeControls.update();
}

function reset3dView() {
  if (sceneData?.camera?.cam2world_gl) {
    applyInitialCameraView();
  } else if (currentModel) {
    frameCameraOnObject(currentModel);
  }
}

function applyInitialCameraView() {
  const matrix = matrix4FromRows(sceneData.camera.cam2world_gl);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  matrix.decompose(position, quaternion, scale);

  const focus = currentModel
    ? new THREE.Box3().setFromObject(currentModel).getCenter(new THREE.Vector3())
    : new THREE.Vector3(0, 0, 0.8);
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion).normalize();
  const distance = Math.max(0.3, forward.dot(focus.clone().sub(position)));

  threeCamera.position.copy(position);
  threeCamera.quaternion.copy(quaternion);
  threeCamera.near = 0.01;
  threeCamera.far = 20;
  threeCamera.updateProjectionMatrix();
  threeControls.target.copy(position.clone().add(forward.multiplyScalar(distance)));
  threeControls.update();
}

function createEnvironment3d() {
  environmentRoot.clear();
  const env = sceneData?.environment || defaultEnvironment();
  const table = env.table;
  const wall = env.wall;
  const tableMaterial = environmentMaterial(table.color, table.texture, 0.55);
  const legMaterial = environmentMaterial(table.color, null, 0.65);
  const wallMaterial = environmentMaterial(wall.color, wall.texture, 0.9);
  const [tx, ty, tz] = table.pose;
  const thickness = table.thickness;

  addBox(environmentRoot, [table.length, table.width, thickness], [tx, ty, tz - thickness / 2], tableMaterial);
  [[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach(([sx, sy]) => {
    const x = tx + sx * (table.length / 2 - 0.05);
    const y = ty + sy * (table.width / 2 - 0.05);
    addBox(environmentRoot, [thickness, thickness, table.height - 0.004], [x, y, table.height / 2 - 0.002], legMaterial);
  });
  addBox(environmentRoot, wall.size, wall.pose, wallMaterial);
}

function addBox(parent, size, position, material) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
  mesh.position.set(...position);
  parent.add(mesh);
}

function environmentMaterial(color, texture, roughness) {
  const material = new THREE.MeshStandardMaterial({color: rgbArrayToColor(color), roughness});
  if (texture) {
    const map = textureLoader.load(`/robotwin_assets/background_texture/${texture}.png`);
    map.wrapS = THREE.RepeatWrapping;
    map.wrapT = THREE.RepeatWrapping;
    map.repeat.set(2, 2);
    material.map = map;
    material.color.set(0xffffff);
  }
  return material;
}

function rgbArrayToColor(color) {
  return new THREE.Color(color[0], color[1], color[2]);
}

function defaultEnvironment() {
  return {
    table: {
      pose: [0, 0, 0.74],
      length: 1.2,
      width: 0.7,
      height: 0.74,
      thickness: 0.05,
      color: [1, 1, 1],
      texture: null,
    },
    wall: {
      pose: [0, 1, 1.5],
      size: [6.0, 1.2, 3.0],
      color: [1, 0.9, 0.9],
      texture: null,
    },
  };
}

function buildMetadata(data) {
  return {
    task_config: data.task_config,
    task_name: data.task_name,
    seed: data.seed,
    pre_grasp_dis: getPreGraspDis(),
    grasp_dis: getGraspDis(),
    perturbation: getPerturbation(),
    cache_hit: data.cache_hit,
    refreshed: data.refreshed,
    camera: data.camera,
    objects: data.objects.map((object) => ({
      name: object.name,
      model_id: object.model_id,
      arm_tag: object.arm_tag,
      pose_world: object.pose_world,
      contact_points: object.contact_points.map((point) => ({
        id: point.id,
        pose_world: point.pose_world,
        grasp_pose_world: point.grasp_pose_world,
        grasp_matrix_world: point.grasp_matrix_world,
        tcp_matrix_world: point.tcp_matrix_world,
        axes_2d: point.axes_2d,
        projection_valid: point.projection_valid,
        in_image: point.in_image,
      })),
    })),
  };
}

function setStatus(text) {
  statusBox.textContent = text;
}

init().catch((error) => setStatus(`初始化失败：${error.message}`));
