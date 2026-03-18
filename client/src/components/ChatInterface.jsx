import { useState, useRef, useEffect } from "react";
import { Send, StopCircle } from "lucide-react";
import { queryStream } from "../services/api";

export default function ChatInterface({ onSourcesUpdate }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");
  const [streaming, setStreaming] = useState(false);
  const esRef    = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendQuery = () => {
    const q = input.trim();
    if (!q || streaming) return;

    // Push user message
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    // Push empty assistant message (will be filled by stream)
    setMessages((prev) => [...prev, { role: "assistant", text: "", streaming: true }]);
    setInput("");
    setStreaming(true);

    esRef.current = queryStream(q, {
      onToken: (token) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last    = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = { ...last, text: last.text + token };
          }
          return updated;
        });
      },
      onDone: () => {
        setMessages((prev) => {
          const updated = [...prev];
          const last    = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = { ...last, streaming: false };
          }
          return updated;
        });
        setStreaming(false);
        esRef.current = null;
      },
      onError: (err) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last    = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = {
              ...last, text: `Error: ${err}`, streaming: false, error: true,
            };
          }
          return updated;
        });
        setStreaming(false);
      },
    });
  };

  const stopStream = () => {
    esRef.current?.close();
    setStreaming(false);
    setMessages((prev) => {
      const updated = [...prev];
      const last    = updated[updated.length - 1];
      if (last?.streaming) {
        updated[updated.length - 1] = { ...last, streaming: false };
      }
      return updated;
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-zinc-600 text-sm font-mono mt-12">
            Ask anything about the loaded file
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`
                max-w-[80%] rounded-xl px-4 py-3 text-sm leading-relaxed
                ${msg.role === "user"
                  ? "bg-zinc-800 text-zinc-100"
                  : msg.error
                    ? "bg-red-950/40 text-red-300 border border-red-800"
                    : "bg-zinc-900 text-zinc-200 border border-zinc-800"
                }
                ${msg.streaming ? "streaming-cursor" : ""}
              `}
            >
              <pre className="whitespace-pre-wrap font-sans">{msg.text || (msg.streaming ? "" : "…")}</pre>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="border-t border-zinc-800 px-4 py-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendQuery()}
          placeholder="Ask a question…"
          disabled={streaming}
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2 text-sm
                     text-zinc-100 placeholder-zinc-600 focus:outline-none focus:border-emerald-600
                     font-mono disabled:opacity-50"
        />
        {streaming ? (
          <button
            onClick={stopStream}
            className="p-2 rounded-lg bg-red-900/40 hover:bg-red-900/70 text-red-400
                       border border-red-800 transition-colors"
            title="Stop generation"
          >
            <StopCircle size={18} />
          </button>
        ) : (
          <button
            onClick={sendQuery}
            disabled={!input.trim()}
            className="p-2 rounded-lg bg-emerald-900/40 hover:bg-emerald-900/70 text-emerald-400
                       border border-emerald-800 transition-colors disabled:opacity-30"
          >
            <Send size={18} />
          </button>
        )}
      </div>
    </div>
  );
}