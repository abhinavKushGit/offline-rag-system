const COLORS = {
  text:  "bg-blue-900/60 text-blue-300 border-blue-700",
  pdf:   "bg-orange-900/60 text-orange-300 border-orange-700",
  image: "bg-purple-900/60 text-purple-300 border-purple-700",
  audio: "bg-green-900/60 text-green-300 border-green-700",
  video: "bg-rose-900/60 text-rose-300 border-rose-700",
};

const ICONS = {
  text:  "TXT",
  pdf:   "PDF",
  image: "IMG",
  audio: "AUD",
  video: "VID",
};

export default function ModalityBadge({ modality, size = "sm" }) {
  if (!modality) return null;
  const cls = COLORS[modality] ?? "bg-zinc-800 text-zinc-400 border-zinc-600";
  const pad = size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm";
  return (
    <span className={`${cls} ${pad} border rounded font-mono font-medium tracking-wider uppercase`}>
      {ICONS[modality] ?? modality}
    </span>
  );
}