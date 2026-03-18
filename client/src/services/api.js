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
 * Upload a file with progress callback.
 * onProgress(0-100), resolves with JSON response.
 */
export function uploadFile(file, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd  = new FormData();
    fd.append("file", file);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          reject(new Error(JSON.parse(xhr.responseText).detail));
        } catch {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      }
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.open("POST", `${BASE}/ingest`);
    xhr.send(fd);
  });
}

/**
 * Open an SSE stream for a query.
 * Calls onToken(str) for each token, onDone() when finished,
 * onError(str) on failure. Returns the EventSource so caller can abort.
 */
export function queryStream(question, { onToken, onDone, onError }) {
  const url = `${BASE}/query?q=${encodeURIComponent(question)}`;
  const es  = new EventSource(url);

  es.onmessage = (e) => {
    try {
      const payload = JSON.parse(e.data);
      if (payload.token !== undefined) onToken(payload.token);
      if (payload.done)  { es.close(); onDone(); }
      if (payload.error) { es.close(); onError(payload.error); }
    } catch {
      onToken(e.data);   // fallback: treat as raw token
    }
  };

  es.onerror = () => { es.close(); onError("Stream connection lost."); };
  return es;
}