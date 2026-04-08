const BASE = "http://localhost:8000";

export async function getStatus() {
  const r = await fetch(`${BASE}/status`);
  if (!r.ok) throw new Error("Backend unreachable");
  return r.json();
}

export async function clearSession() {
  const r = await fetch(`${BASE}/session`, { method: "DELETE" });
  return r.json();
}

/**
 * Upload a file and poll /status until ingestion completes.
 *
 * onProgress(0-100)  — upload byte progress
 * onStage(msg)       — human-readable status updates while processing
 * resolves with the final /status payload when pipeline_ready === true
 * rejects on upload error or backend error
 */
export function uploadFile(file, onProgress = () => {}, onStage = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd  = new FormData();
    fd.append("file", file);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };

    xhr.onload = () => {
      if (xhr.status === 202) {
        // Upload accepted — now poll until done
        onStage("Processing… this may take a few minutes for video/audio.");
        _pollUntilReady(resolve, reject, onStage);
      } else {
        try   { reject(new Error(JSON.parse(xhr.responseText).detail)); }
        catch { reject(new Error(`Upload failed: ${xhr.status}`)); }
      }
    };

    xhr.onerror   = () => reject(new Error("Network error during upload."));
    xhr.ontimeout = () => reject(new Error("Upload request timed out."));

    xhr.open("POST", `${BASE}/ingest`);
    xhr.send(fd);
  });
}

/**
 * Poll /status every 3 seconds until pipeline_ready or error.
 */
function _pollUntilReady(resolve, reject, onStage, attempt = 0) {
  setTimeout(async () => {
    try {
      const status = await getStatus();

      if (status.error) {
        reject(new Error(status.error));
        return;
      }

      if (status.pipeline_ready) {
        resolve(status);
        return;
      }

      if (status.processing) {
        const dots = ".".repeat((attempt % 3) + 1);
        onStage(`Processing${dots}`);
        _pollUntilReady(resolve, reject, onStage, attempt + 1);
        return;
      }

      // processing flipped to false but not ready — something went wrong silently
      reject(new Error("Ingestion ended without a ready pipeline."));
    } catch (err) {
      reject(err);
    }
  }, 3000); // poll every 3 s
}

/**
 * Open an SSE stream for a query.
 * onToken(str), onSources(arr), onDone(), onError(str)
 * Returns the EventSource so caller can abort.
 */
export function queryStream(question, { onToken, onSources, onDone, onError }) {
  const url = `${BASE}/query?q=${encodeURIComponent(question)}`;
  const es  = new EventSource(url);

  es.onmessage = (e) => {
    try {
      const payload = JSON.parse(e.data);
      if (payload.sources !== undefined) { onSources && onSources(payload.sources); return; }
      if (payload.token   !== undefined) { onToken(payload.token); return; }
      if (payload.done)  { es.close(); onDone();          return; }
      if (payload.error) { es.close(); onError(payload.error); return; }
    } catch {
      onToken(e.data);
    }
  };

  es.onerror = () => { es.close(); onError("Stream connection lost."); };
  return es;
}