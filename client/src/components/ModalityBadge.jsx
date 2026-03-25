const COLORS = {
  text:  "bg-sky-950 text-sky-300 border-sky-800",
  pdf:   "bg-amber-950 text-amber-300 border-amber-800",
  image: "bg-violet-950 text-violet-300 border-violet-800",
  audio: "bg-emerald-950 text-emerald-300 border-emerald-800",
  video: "bg-rose-950 text-rose-300 border-rose-800",
};

const LABELS = { text:"TXT", pdf:"PDF", image:"IMG", audio:"AUD", video:"VID" };

export default function ModalityBadge({ modality, size = "sm" }) {
  if (!modality) return null;
  const cls = COLORS[modality] ?? "bg-zinc-800 text-zinc-400 border-zinc-700";
  const pad = size === "sm" ? "px-1.5 py-0.5 text-[10px]"
            : size === "md" ? "px-2 py-0.5 text-xs"
            :                 "px-2.5 py-1 text-xs";
  return (
    <span className={`${cls} ${pad} border rounded font-mono font-medium tracking-widest uppercase inline-block`}>
      {LABELS[modality] ?? modality}
    </span>
  );
}