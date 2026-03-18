import { useState, useEffect } from "react";
import StatusBar    from "./components/StatusBar";
import FileUpload   from "./components/FileUpload";
import ChatInterface from "./components/ChatInterface";
import { getStatus } from "./services/api";

export default function App() {
  const [status, setStatus] = useState({
    pipeline_ready:   false,
    current_file:     null,
    current_modality: null,
    processing:       false,
  });

  // Poll status every 3s (covers async processing state)
  useEffect(() => {
    const refresh = async () => {
      try { setStatus(await getStatus()); } catch {}
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
    });
  };

  const handleCleared = () => {
    setStatus({
      pipeline_ready:   false,
      current_file:     null,
      current_modality: null,
      processing:       false,
    });
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <StatusBar status={status} onCleared={handleCleared} />

      <main className="flex-1 overflow-hidden">
        {status.pipeline_ready ? (
          <ChatInterface />
        ) : (
          <FileUpload onIngested={handleIngested} />
        )}
      </main>
    </div>
  );
}