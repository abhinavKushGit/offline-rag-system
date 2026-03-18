import { useState, useRef } from "react";
import { UploadCloud, Loader } from "lucide-react";
import { uploadFile } from "../services/api";
import ModalityBadge from "./ModalityBadge";

const MODALITY_MAP = {
  ".txt": "text", ".pdf": "pdf",
  ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
  ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
  ".mp4": "video", ".mkv": "video", ".mov": "video",
};

function detectModality(filename) {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  return MODALITY_MAP[ext] ?? null;
}

export default function FileUpload({ onIngested }) {
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const inputRef = useRef();

  const handleFile = async (file) => {
    setError(null);
    const modality = detectModality(file.name);
    if (!modality) {
      setError(`Unsupported file type: ${file.name.split(".").pop()}`);
      return;
    }
    setPreview({ name: file.name, modality, size: (file.size / 1024 / 1024).toFixed(1) });
    setLoading(true);
    setProgress(0);
    try {
      const result = await uploadFile(file, setProgress);
      onIngested(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !loading && inputRef.current.click()}
        className={`
          w-full max-w-lg border-2 border-dashed rounded-xl p-12
          flex flex-col items-center gap-4 cursor-pointer transition-all duration-200
          ${dragging
            ? "border-emerald-500 bg-emerald-950/20"
            : "border-zinc-700 hover:border-zinc-500 bg-zinc-900/40"
          }
          ${loading ? "pointer-events-none opacity-70" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept=".txt,.pdf,.png,.jpg,.jpeg,.webp,.mp3,.wav,.m4a,.mp4,.mkv,.mov"
          onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
        />

        {loading ? (
          <Loader size={36} className="text-emerald-400 animate-spin" />
        ) : (
          <UploadCloud size={36} className="text-zinc-500" />
        )}

        {preview && !loading ? (
          <div className="flex items-center gap-2">
            <ModalityBadge modality={preview.modality} size="md" />
            <span className="text-sm text-zinc-300 font-mono">{preview.name}</span>
            <span className="text-xs text-zinc-600">{preview.size} MB</span>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-zinc-300 text-sm">
              {loading ? "Processing file…" : "Drop a file or click to browse"}
            </p>
            <p className="text-zinc-600 text-xs mt-1">
              txt · pdf · png · jpg · mp3 · wav · mp4 · mkv
            </p>
          </div>
        )}

        {loading && (
          <div className="w-full bg-zinc-800 rounded-full h-1.5">
            <div
              className="bg-emerald-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>

      {/* Ingestion status hint */}
      {loading && (
        <p className="text-xs text-zinc-500 font-mono animate-pulse">
          {preview?.modality === "video"
            ? "Transcribing audio + captioning keyframes… (may take a few minutes)"
            : preview?.modality === "audio"
            ? "Transcribing audio…"
            : preview?.modality === "image"
            ? "Generating visual caption…"
            : "Chunking and indexing…"}
        </p>
      )}

      {error && (
        <p className="text-red-400 text-sm font-mono bg-red-950/30 border border-red-800 rounded px-4 py-2">
          ✗ {error}
        </p>
      )}
    </div>
  );
}