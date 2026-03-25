import { useState, useEffect } from "react";
import Sidebar       from "./components/Sidebar";
import FileUpload    from "./components/FileUpload";
import ChatInterface from "./components/ChatInterface";
import { getStatus } from "./services/api";

export default function App() {
  const [status, setStatus] = useState({
    pipeline_ready:   false,
    current_file:     null,
    current_modality: null,
    processing:       false,
    previewUrl:       null,
  });

  // Poll backend status every 3s — preserve previewUrl (backend doesn't know it)
  useEffect(() => {
    const refresh = async () => {
      try {
        const s = await getStatus();
        setStatus(prev => ({ ...s, previewUrl: prev.previewUrl }));
      } catch {}
    };
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, []);

  const handleIngested = (result) => {
    setStatus({
      pipeline_ready:   true,
      current_file:     result.filename,
      current_modality: result.modality,
      processing:       false,
      previewUrl:       result.previewUrl ?? null,
    });
  };

  const handleCleared = () => {
    if (status.previewUrl) URL.revokeObjectURL(status.previewUrl);
    setStatus({
      pipeline_ready:   false,
      current_file:     null,
      current_modality: null,
      processing:       false,
      previewUrl:       null,
    });
  };

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950">

      {/* Sidebar */}
      <Sidebar status={status} onCleared={handleCleared} />

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden dot-grid">
        {status.pipeline_ready ? (
          <>
            <header className="shrink-0 flex items-center justify-center px-6 py-4
                               border-b border-zinc-800/60 bg-zinc-950/80 backdrop-blur">
              <div className="text-center">
                <h1 className="font-display text-2xl font-bold text-zinc-100 tracking-tight leading-none">
                  OmniRAG
                </h1>
                <p className="text-[11px] font-mono text-zinc-600 mt-0.5">
                  Ask anything · Retrieval-Augmented Generation
                </p>
              </div>
            </header>
            <div className="flex-1 overflow-hidden">
              <ChatInterface />
            </div>
          </>
        ) : (
          <FileUpload onIngested={handleIngested} />
        )}
      </main>
    </div>
  );
}