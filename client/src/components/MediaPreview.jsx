import { Music, FileText, File } from "lucide-react";

export default function MediaPreview({ previewUrl, modality, filename }) {
  if (!modality) return null;

  return (
    <div className="px-3 py-3 border-b border-zinc-800/60">
      <p className="text-[10px] font-mono text-zinc-600 uppercase tracking-widest mb-2 px-2">
        Loaded File
      </p>
      <div className="rounded-xl overflow-hidden border border-zinc-800 bg-zinc-900">

        {/* IMAGE */}
        {modality === "image" && previewUrl && (
          <img src={previewUrl} alt={filename}
               className="w-full object-cover max-h-36" />
        )}

        {/* VIDEO */}
        {modality === "video" && previewUrl && (
          <video src={previewUrl} className="w-full max-h-36 object-cover"
                 muted playsInline preload="metadata"
                 onLoadedMetadata={(e) => { e.target.currentTime = 0.5; }} />
        )}

        {/* AUDIO */}
        {modality === "audio" && previewUrl && (
          <div className="flex flex-col items-center gap-2 px-3 py-4">
            <div className="w-10 h-10 rounded-full bg-emerald-950/40 border border-emerald-900
                            flex items-center justify-center">
              <Music size={16} className="text-emerald-500" />
            </div>
            <audio src={previewUrl} controls className="w-full h-7 opacity-80" />
          </div>
        )}

        {/* PDF */}
        {modality === "pdf" && (
          <div className="flex items-center gap-2.5 px-3 py-4">
            <div className="w-10 h-10 rounded-xl bg-amber-950/40 border border-amber-900
                            flex items-center justify-center shrink-0">
              <FileText size={16} className="text-amber-500" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-mono text-zinc-300 truncate">{filename}</p>
              <p className="text-[10px] font-mono text-zinc-600 mt-0.5">PDF Document</p>
            </div>
          </div>
        )}

        {/* TEXT */}
        {modality === "text" && (
          <div className="flex items-center gap-2.5 px-3 py-4">
            <div className="w-10 h-10 rounded-xl bg-sky-950/40 border border-sky-900
                            flex items-center justify-center shrink-0">
              <File size={16} className="text-sky-400" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-mono text-zinc-300 truncate">{filename}</p>
              <p className="text-[10px] font-mono text-zinc-600 mt-0.5">Text File</p>
            </div>
          </div>
        )}

        {/* Filename caption for image/video */}
        {["image", "video"].includes(modality) && (
          <div className="px-3 py-1.5 bg-zinc-950/60">
            <p className="text-[10px] font-mono text-zinc-600 truncate">{filename}</p>
          </div>
        )}
      </div>
    </div>
  );
}