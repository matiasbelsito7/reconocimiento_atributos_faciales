/**
 * Modulo de camara web con captura manual.
 */

let stream = null;
let videoElement = null;
let canvasElement = null;

export function isCameraSupported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

export async function startCamera(video, canvas) {
    videoElement = video;
    canvasElement = canvas;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user",
            },
            audio: false,
        });
        videoElement.srcObject = stream;
        await videoElement.play();
        return true;
    } catch (err) {
        console.error("Camera access denied:", err);
        return false;
    }
}

export function captureFrame() {
    if (!videoElement || !canvasElement) return null;

    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    const ctx = canvasElement.getContext("2d");
    ctx.drawImage(videoElement, 0, 0);

    return new Promise((resolve) => {
        canvasElement.toBlob(
            (blob) => resolve(blob),
            "image/jpeg",
            0.9
        );
    });
}

export function stopCamera() {
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
    if (videoElement) {
        videoElement.srcObject = null;
    }
}
