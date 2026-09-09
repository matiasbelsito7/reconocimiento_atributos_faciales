/**
 * App principal - orquesta camara, upload y prediccion.
 */

import { isCameraSupported, startCamera, captureFrame, stopCamera } from "./webcam.js";
import { setupUpload, validateFile, fileToFormData, fileToPreviewUrl } from "./upload.js";

const API_BASE = "/api";

/* DOM elements */
const serviceStatus = document.getElementById("service-status");
const webcamSection = document.getElementById("webcam-section");
const uploadSection = document.getElementById("upload-section");
const previewSection = document.getElementById("preview-section");
const loadingIndicator = document.getElementById("loading-indicator");
const errorMessage = document.getElementById("error-message");
const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsContent = document.getElementById("results-content");

const webcamVideo = document.getElementById("webcam-video");
const webcamCanvas = document.getElementById("webcam-canvas");
const btnCapture = document.getElementById("btn-capture");
const btnStopCamera = document.getElementById("btn-stop-camera");
const btnStartCamera = document.getElementById("btn-start-camera");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

const previewImage = document.getElementById("preview-image");
const btnPredict = document.getElementById("btn-predict");
const btnClear = document.getElementById("btn-clear");

const resultFaces = document.getElementById("result-faces");
const resultTime = document.getElementById("result-time");
const resultsImage = document.getElementById("results-image");
const resultsCanvas = document.getElementById("results-canvas");
const faceTabs = document.getElementById("face-tabs");
const attributesGrid = document.getElementById("attributes-grid");

let currentFile = null;
let attributes = [];
let currentResult = null;
let activeFaceIndex = 0;

/* --- Init --- */
async function init() {
    setupUpload(dropZone, fileInput, handleFile);
    setupCameraButtons();
    setupPreviewButtons();
    await checkHealth();
    await loadAttributes();
}

/* --- Health check --- */
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === "ok") {
            serviceStatus.textContent = data.model_loaded
                ? "Servicio activo"
                : "Servicio activo (sin modelo cargado)";
            serviceStatus.className = "status-badge " + (data.model_loaded ? "status-ok" : "status-error");
        } else {
            throw new Error("Status not ok");
        }
    } catch {
        serviceStatus.textContent = "Servicio no disponible";
        serviceStatus.className = "status-badge status-error";
    }
}

/* --- Load attributes --- */
async function loadAttributes() {
    try {
        const res = await fetch(`${API_BASE}/attributes`);
        const data = await res.json();
        attributes = data.attributes;
    } catch {
        attributes = [];
    }
}

/* --- Camera --- */
function setupCameraButtons() {
    btnStartCamera.addEventListener("click", async () => {
        if (!isCameraSupported()) {
            showError("La camara no es compatible con este navegador.");
            return;
        }
        const ok = await startCamera(webcamVideo, webcamCanvas);
        if (ok) {
            uploadSection.style.display = "none";
            webcamSection.style.display = "";
            btnCapture.disabled = false;
        } else {
            showError("No se pudo acceder a la camara. Verifica los permisos.");
        }
    });

    btnCapture.addEventListener("click", async () => {
        const blob = await captureFrame();
        if (blob) {
            const file = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
            handleFile(file);
            stopCamera();
            webcamSection.style.display = "none";
            uploadSection.style.display = "";
            btnCapture.disabled = true;
        }
    });

    btnStopCamera.addEventListener("click", () => {
        stopCamera();
        webcamSection.style.display = "none";
        uploadSection.style.display = "";
        btnCapture.disabled = true;
    });
}

/* --- File handling --- */
function handleFile(file) {
    const validation = validateFile(file);
    if (!validation.valid) {
        showError(validation.error);
        return;
    }

    hideError();
    currentFile = file;
    previewImage.src = fileToPreviewUrl(file);
    uploadSection.style.display = "none";
    previewSection.style.display = "";
}

/* --- Preview controls --- */
function setupPreviewButtons() {
    btnPredict.addEventListener("click", predict);
    btnClear.addEventListener("click", clearAll);
}

async function predict() {
    if (!currentFile) return;

    hideError();
    showLoading(true);

    try {
        const formData = fileToFormData(currentFile);
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Error del servidor" }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        showResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

/* --- Results --- */
function showResults(data) {
    hideError();
    resultsPlaceholder.style.display = "none";
    resultsContent.style.display = "";

    resultFaces.textContent = `Rostros detectados: ${data.num_faces_detected}`;
    resultTime.textContent = `Tiempo: ${data.inference_time_ms.toFixed(0)}ms`;

    resultsImage.src = previewImage.src;

    attributesGrid.innerHTML = "";
    faceTabs.innerHTML = "";
    currentResult = data;

    if (data.error) {
        showError(data.error);
        return;
    }

    if (data.faces.length === 0) {
        showError("No se detectaron rostros en la imagen.");
        return;
    }

    activeFaceIndex = findLargestFaceIndex(data.faces);
    renderFaceTabs(data.faces.length);
    resultsImage.onload = drawBoundingBoxes;
    drawBoundingBoxes();
    renderFaceAttributes(activeFaceIndex);
}

function findLargestFaceIndex(faces) {
    let largestIndex = 0;
    let largestArea = -1;
    for (let i = 0; i < faces.length; i++) {
        const area = faces[i].bbox.w * faces[i].bbox.h;
        if (area > largestArea) {
            largestArea = area;
            largestIndex = i;
        }
    }
    return largestIndex;
}

function renderFaceTabs(count) {
    faceTabs.innerHTML = "";
    if (count <= 1) return;

    for (let i = 0; i < count; i++) {
        const tab = document.createElement("button");
        tab.className = i === activeFaceIndex ? "face-tab active" : "face-tab";
        tab.textContent = `Cara ${i + 1}`;
        tab.addEventListener("click", () => {
            activeFaceIndex = i;
            renderFaceTabs(count);
            drawBoundingBoxes();
            renderFaceAttributes(i);
        });
        faceTabs.appendChild(tab);
    }
}

function drawBoundingBoxes() {
    if (!currentResult || !resultsImage.naturalWidth) return;

    const canvas = resultsCanvas;
    const ctx = canvas.getContext("2d");
    const imgW = resultsImage.clientWidth;
    const imgH = resultsImage.clientHeight;

    canvas.width = imgW;
    canvas.height = imgH;
    ctx.clearRect(0, 0, imgW, imgH);

    const scale = Math.min(
        imgW / resultsImage.naturalWidth,
        imgH / resultsImage.naturalHeight
    );
    const offsetX = (imgW - resultsImage.naturalWidth * scale) / 2;
    const offsetY = (imgH - resultsImage.naturalHeight * scale) / 2;

    currentResult.faces.forEach((face, i) => {
        const box = face.bbox;
        const x = offsetX + box.x * scale;
        const y = offsetY + box.y * scale;
        const w = box.w * scale;
        const h = box.h * scale;
        const isActive = i === activeFaceIndex;

        ctx.strokeStyle = isActive ? "#6366f1" : "rgba(139, 143, 163, 0.6)";
        ctx.lineWidth = isActive ? 3 : 1.5;
        ctx.strokeRect(x, y, w, h);

        const label = `${i + 1}`;
        const labelH = isActive ? 20 : 16;
        const padX = 5;
        ctx.font = `bold ${isActive ? 12 : 10}px sans-serif`;
        const textW = ctx.measureText(label).width;
        const labelX = x;
        const labelY = y - labelH - 2 > 0 ? y - labelH - 2 : y;
        ctx.fillStyle = isActive ? "#6366f1" : "rgba(139, 143, 163, 0.85)";
        ctx.fillRect(labelX, labelY, textW + padX * 2, labelH);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, labelX + (textW + padX * 2) / 2, labelY + labelH / 2);
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
    });
}

function renderFaceAttributes(faceIndex) {
    const face = currentResult.faces[faceIndex];
    attributesGrid.innerHTML = "";

    const sorted = Object.entries(face.attributes).sort((a, b) => b[1] - a[1]);

    for (const [name, score] of sorted) {
        const card = document.createElement("div");
        card.className = "attr-card";

        const displayName = getDisplayName(name);
        const scoreClass = score >= 0.7 ? "attr-high" : score >= 0.3 ? "attr-medium" : "attr-low";

        const decisionInfo = face.attribute_decisions ? face.attribute_decisions[name] : null;
        let badgeHtml = "";
        if (decisionInfo) {
            const label = decisionInfo.decision === "si"
                ? "Sí"
                : decisionInfo.decision === "no"
                    ? "No"
                    : "?";
            const detail = `Umbral ${(decisionInfo.threshold * 100).toFixed(0)}% · Margen \u00b1${(decisionInfo.margin * 100).toFixed(0)}%`;
            badgeHtml = `<span class="attr-decision attr-decision-${decisionInfo.decision}" title="${detail}">${label}</span>`;
        }

        card.innerHTML = `
            <span class="attr-name" title="${name}">${displayName}</span>
            <span class="attr-right">
                ${badgeHtml}
                <span class="attr-score ${scoreClass}">${(score * 100).toFixed(0)}%</span>
            </span>
        `;
        attributesGrid.appendChild(card);
    }
}

function getDisplayName(name) {
    const attr = attributes.find((a) => a.name === name);
    return attr ? attr.display_name : name.replace(/_/g, " ");
}

/* --- UI helpers --- */
function showLoading(show) {
    loadingIndicator.style.display = show ? "" : "none";
    btnPredict.disabled = show;
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.style.display = "";
}

function hideError() {
    errorMessage.style.display = "none";
}

function clearAll() {
    currentFile = null;
    currentResult = null;
    activeFaceIndex = 0;
    previewImage.src = "";
    previewSection.style.display = "none";
    uploadSection.style.display = "";
    resultsContent.style.display = "none";
    resultsPlaceholder.style.display = "";
    faceTabs.innerHTML = "";
    hideError();
}

window.addEventListener("resize", () => {
    if (currentResult && resultsContent.style.display !== "none") {
        drawBoundingBoxes();
    }
});

/* --- Boot --- */
init();
