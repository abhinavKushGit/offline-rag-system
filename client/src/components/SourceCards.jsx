import ModalityBadge from "./ModalityBadge";
import { FileText, Clock, Hash, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);

  const formatTime = (t) => {
    const s = Math.floor(Number(t));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  return (
    <div className="source-card bg-zinc-900/60 border border-zinc-800 rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-800/60">
        <span className="font-mono text-[10px] text-zinc-600 w-4 shrink-0">#{index + 1}</span>
        <ModalityBadge modality={source.modality} size="sm" />
        <span className="font-mono text-xs text-zinc-400 truncate flex-1">
          {source.source || "unknown"}
        </span>
        <div className="flex items-center gap-2 shrink-0">
          {source.start_time != null && (
            <span className="flex items-center gap-1 text-[10px] font-mono text-zinc-500">
              <Clock size={9} />{formatTime(source.start_time)}
            </span>
          )}
          {source.page != null && (
            <span className="flex items-center gap-1 text-[10px] font-mono text-zinc-500">
              <Hash size={9} />p.{source.page}
            </span>
          )}
          {source.score > 0 && (
            <span className="text-[10px] font-mono text-emerald-700">
              {(source.score * 100).toFixed(0)}%
            </span>
          )}
          <button onClick={() => setExpanded(!expanded)}
            className="text-zinc-600 hover:text-zinc-400 transition-colors">
            {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
        </div>
      </div>

      {source.section && source.section !== "General" && (
        <div className="px-3 pt-1.5">
          <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider">
            § {source.section}
          </span>
        </div>
      )}

      <div className="px-3 py-2">
        <p className={`text-xs text-zinc-500 leading-relaxed ${expanded ? "" : "line-clamp-2"}`}
           style={{ fontFamily: "inherit" }}>
          {source.text}
        </p>
      </div>
    </div>
  );
}

export default function SourceCards({ sources }) {
  const [open, setOpen] = useState(true);
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2 fade-in">
      <button onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-[11px] font-mono text-zinc-600
                   hover:text-zinc-400 transition-colors mb-2">
        <FileText size={11} />
        {sources.length} source{sources.length !== 1 ? "s" : ""} retrieved
        {open ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
      </button>

      {open && (
        <div className="grid grid-cols-1 gap-1.5">
          {sources.map((s, i) => <SourceCard key={i} source={s} index={i} />)}
        </div>
      )}
    </div>
  );
}