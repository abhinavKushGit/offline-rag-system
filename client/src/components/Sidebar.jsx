import { Cpu, FileText, Volume2, Image, Video, AlignLeft,
         Layers, Zap, Database, Trash2 } from "lucide-react";
import ModalityBadge  from "./ModalityBadge";
import MediaPreview   from "./MediaPreview";
import { clearSession } from "../services/api";

const MODALITY_ICONS = {
  text:  <AlignLeft size={13} />,
  pdf:   <FileText  size={13} />,
  audio: <Volume2   size={13} />,
  image: <Image     size={13} />,
  video: <Video     size={13} />,
};

const STACK = [
  { icon: <Layers   size={12} />, label: "5-modality RAG" },
  { icon: <Zap      size={12} />, label: "SSE token streaming" },
  { icon: <Database size={12} />, label: "FAISS + CLIP dual index" },
  { icon: <Cpu      size={12} />, label: "Phi-3 · 20 GPU layers" },
];

export default function Sidebar({ status, onCleared }) {
  const handleClear = async () => {
    await clearSession();
    onCleared();
  };

  return (
    <aside className="w-56 shrink-0 h-full flex flex-col border-r border-zinc-800/60
                      sidebar-glow bg-zinc-950 overflow-y-auto">

      {/* Brand */}
      <div className="px-5 pt-6 pb-4 border-b border-zinc-800/60 shrink-0">
        <div className="flex items-center gap-2 mb-1">
          <Cpu size={15} className="text-emerald-500" />
          <span className="font-display text-base font-bold text-zinc-100 tracking-tight">
            OmniRAG
          </span>
          <span className="font-mono text-[10px] text-zinc-700 mt-0.5">v6</span>
        </div>
        <p className="text-[11px] font-mono text-zinc-600 leading-relaxed">
          Offline Multimodal<br />RAG System
        </p>
      </div>

      {/* Media preview */}
      {status.pipeline_ready && (
        <MediaPreview
          previewUrl={status.previewUrl}
          modality={status.current_modality}
          filename={status.current_file}
        />
      )}

      {/* Session */}
      <div className="px-5 py-4 border-b border-zinc-800/60 shrink-0">
        <p className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-3">
          Session
        </p>
        {status.pipeline_ready ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              {MODALITY_ICONS[status.current_modality]}
              <ModalityBadge modality={status.current_modality} size="sm" />
            </div>
            <p className="text-xs font-mono text-zinc-400 truncate" title={status.current_file}>
              {status.current_file}
            </p>
            <div className="flex items-center gap-1.5 mt-0.5">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[11px] font-mono text-emerald-600">Pipeline ready</span>
            </div>
            <button onClick={handleClear}
              className="flex items-center gap-1.5 text-[11px] font-mono text-zinc-600
                         hover:text-red-400 transition-colors mt-1">
              <Trash2 size={11} /> Clear session
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-zinc-700" />
              <span className="text-[11px] font-mono text-zinc-600">
                {status.processing ? "Processing…" : "No file loaded"}
              </span>
            </div>
            {status.processing && (
              <div className="w-full bg-zinc-800 rounded-full h-0.5 overflow-hidden">
                <div className="bg-emerald-600 h-0.5 rounded-full animate-pulse w-2/3" />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modalities */}
      <div className="px-5 py-4 border-b border-zinc-800/60 shrink-0">
        <p className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-3">
          Modalities
        </p>
        <div className="flex flex-col gap-1.5">
          {[
            { m: "text",  label: "Plain text files" },
            { m: "pdf",   label: "PDF documents"    },
            { m: "image", label: "Images"           },
            { m: "audio", label: "Audio files"      },
            { m: "video", label: "Video files"      },
          ].map(({ m, label }) => (
            <div key={m}
              className={`flex items-center gap-2 rounded-lg px-2 py-1.5 transition-opacity
                ${status.current_modality === m
                  ? "bg-zinc-800/60 border border-zinc-700 opacity-100"
                  : "opacity-40"}`}>
              <ModalityBadge modality={m} size="sm" />
              <span className="text-[11px] text-zinc-500">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Stack */}
      <div className="px-5 py-4 shrink-0">
        <p className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-3">
          Stack
        </p>
        <div className="flex flex-col gap-1.5">
          {STACK.map((c, i) => (
            <div key={i} className="flex items-center gap-2 text-zinc-600">
              <span className="text-zinc-700">{c.icon}</span>
              <span className="text-[11px] font-mono">{c.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-auto px-5 py-4 border-t border-zinc-800/60 shrink-0">
        <p className="text-[10px] font-mono text-zinc-700 leading-relaxed">
          RTX 3050 · 4GB VRAM<br />
          CUDA 12.0 · Python 3.12<br />
          Fully offline · No cloud
        </p>
      </div>
    </aside>
  );
}