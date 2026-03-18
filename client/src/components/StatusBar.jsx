import { Cpu, Trash2 } from "lucide-react";
import ModalityBadge from "./ModalityBadge";
import { clearSession } from "../services/api";

export default function StatusBar({ status, onCleared }) {
  const handleClear = async () => {
    await clearSession();
    onCleared();
  };

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-10">
      <div className="flex items-center gap-3">
        <Cpu size={18} className="text-emerald-400" />
        <span className="font-mono text-sm font-medium text-zinc-100 tracking-tight">
          OmniRAG
        </span>
        <span className="text-zinc-600 text-xs font-mono">v6</span>
      </div>

      <div className="flex items-center gap-3">
        {status.pipeline_ready ? (
          <>
            <ModalityBadge modality={status.current_modality} />
            <span className="text-zinc-400 text-xs font-mono truncate max-w-[200px]">
              {status.current_file}
            </span>
            <button
              onClick={handleClear}
              className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-red-400 transition-colors"
              title="Clear session"
            >
              <Trash2 size={13} />
              Clear
            </button>
          </>
        ) : (
          <span className="text-zinc-600 text-xs font-mono">
            {status.processing ? "Processing…" : "No file loaded"}
          </span>
        )}
      </div>
    </header>
  );
}