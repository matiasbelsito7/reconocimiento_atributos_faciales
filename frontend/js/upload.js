/**
 * Modulo de subida de archivos con drag & drop.
 */

const MAX_SIZE_MB = 10;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export function setupUpload(dropZone, fileInput, onFileSelected) {
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) onFileSelected(file);
    });

    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) onFileSelected(file);
        fileInput.value = "";
    });
}

export function validateFile(file) {
    if (!ALLOWED_TYPES.includes(file.type)) {
        return { valid: false, error: "Tipo de archivo no soportado. Usa JPEG, PNG, WebP o BMP." };
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        return { valid: false, error: `El archivo supera el limite de ${MAX_SIZE_MB}MB.` };
    }
    return { valid: true };
}

export function fileToFormData(file) {
    const formData = new FormData();
    formData.append("file", file, file.name);
    return formData;
}

export function fileToPreviewUrl(file) {
    return URL.createObjectURL(file);
}
