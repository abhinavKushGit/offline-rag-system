import { useState, useRef } from "react";
import { UploadCloud, Loader } from "lucide-react";
import { uploadFile } from "../services/api";
import ModalityBadge from "./ModalityBadge";

const MODALITY_MAP = {
  ".txt":  "text",  ".pdf":  "pdf",
  ".png":  "image", ".jpg":  "image", ".jpeg": "image",
  ".webp": "image", ".bmp":  "image",
  ".mp3":  "audio", ".wav":  "audio", ".m4a":  "audio", ".flac": "audio",
  ".mp4":  "video", ".mkv":  "video", ".mov":  "video", ".avi":  "video",
};

const HINTS = {
  video: "Transcribing audio + captioning keyframes… may take a few minutes",
  audio: "Transcribing audio with Whisper…",
  image: "Generating visual caption with Qwen2-VL…",
  pdf:   "Extracting and chunking pages…",
  text:  "Chunking and indexing…",
};

function detectModality(filename) {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  return MODALITY_MAP[ext] ?? null;
}

export default function FileUpload({ onIngested }) {
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);
  const [preview,  setPreview]  = useState(null);
  const inputRef = useRef();

  const handleFile = async (file) => {
    setError(null);
    const modality = detectModality(file.name);
    if (!modality) {
      setError(`Unsupported format: .${file.name.split(".").pop()}`);
      return;
    }

    const previewUrl = ["image", "video", "audio"].includes(modality)
      ? URL.createObjectURL(file) : null;

    setPreview({ name: file.name, modality, size: (file.size / 1024 / 1024).toFixed(1) });
    setLoading(true);
    setProgress(0);

    try {
      const result = await uploadFile(file, setProgress);
      onIngested({ ...result, previewUrl, previewModality: modality });
    } catch (e) {
      setError(e.message);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
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
    <div className="flex flex-col items-center justify-center h-full gap-8 px-6">

      {/* Title */}
      <div className="text-center fade-in">
        <h1 className="font-display text-5xl font-bold text-zinc-100 tracking-tight leading-none mb-3">
          OmniRAG
        </h1>
        <p className="text-zinc-500 text-sm font-mono">
          Offline · Multimodal · Retrieval-Augmented Generation
        </p>
        <div className="flex items-center justify-center gap-3 mt-3">
          {["TEXT","PDF","IMAGE","AUDIO","VIDEO"].map(m => (
            <span key={m} className="text-[10px] font-mono text-zinc-700 tracking-widest">{m}</span>
          ))}
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !loading && inputRef.current.click()}
        className={`
          relative w-full max-w-md border border-dashed rounded-2xl p-10
          flex flex-col items-center gap-4 cursor-pointer transition-all duration-300
          ${dragging ? "border-emerald-500 bg-emerald-950/10 scale-[1.01]"
                     : "border-zinc-700 hover:border-zinc-500 bg-zinc-900/30 hover:bg-zinc-900/50"}
          ${loading ? "pointer-events-none" : ""}
        `}
      >
        <input ref={inputRef} type="file" className="hidden"
          accept=".txt,.pdf,.png,.jpg,.jpeg,.webp,.bmp,.mp3,.wav,.m4a,.flac,.mp4,.mkv,.mov,.avi"
          onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])} />

        <div className={`w-14 h-14 rounded-2xl border flex items-center justify-center transition-colors
          ${loading ? "border-emerald-800 bg-emerald-950/30" : "border-zinc-700 bg-zinc-900"}`}>
          {loading
            ? <Loader size={24} className="text-emerald-400 animate-spin" />
            : <UploadCloud size={24} className="text-zinc-500" />}
        </div>

        {preview ? (
          <div className="flex flex-col items-center gap-1.5 text-center">
            <ModalityBadge modality={preview.modality} size="md" />
            <span className="text-sm text-zinc-300 font-mono">{preview.name}</span>
            <span className="text-xs text-zinc-600">{preview.size} MB</span>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-zinc-300 text-sm mb-1">
              Drop a file or{" "}
              <span className="text-emerald-400 underline underline-offset-2">browse</span>
            </p>
            <p className="text-zinc-600 text-xs font-mono">
              txt · pdf · png · jpg · mp3 · wav · mp4 · mkv
            </p>
          </div>
        )}

        {loading && (
          <div className="w-full bg-zinc-800 rounded-full h-0.5 overflow-hidden">
            <div className="bg-emerald-500 h-0.5 rounded-full transition-all duration-300"
                 style={{ width: `${progress}%` }} />
          </div>
        )}
      </div>

      {loading && preview && (
        <p className="text-xs font-mono text-zinc-500 animate-pulse text-center max-w-sm fade-in">
          {HINTS[preview.modality] ?? "Processing…"}
        </p>
      )}

      {error && (
        <div className="flex items-center gap-2 text-sm font-mono text-red-400
                        bg-red-950/30 border border-red-900 rounded-xl px-4 py-2.5 fade-in">
          <span className="text-red-600">✗</span>{error}
        </div>
      )}
    </div>
  );
}