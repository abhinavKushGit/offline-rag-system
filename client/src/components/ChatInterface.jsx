import { useState, useRef, useEffect } from "react";
import { Send, StopCircle, Bot, User } from "lucide-react";
import { queryStream } from "../services/api";
import SourceCards from "./SourceCards";

export default function ChatInterface() {
  const [messages,  setMessages]  = useState([]);
  const [input,     setInput]     = useState("");
  const [streaming, setStreaming] = useState(false);
  const esRef     = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendQuery = () => {
    const q = input.trim();
    if (!q || streaming) return;

    setMessages(prev => [
      ...prev,
      { role: "user",      text: q },
      { role: "assistant", text: "", streaming: true, sources: null },
    ]);
    setInput("");
    setStreaming(true);

    esRef.current = queryStream(q, {
      onToken: (token) => {
        setMessages(prev => {
          const u = [...prev];
          const last = u[u.length - 1];
          if (last?.role === "assistant") u[u.length - 1] = { ...last, text: last.text + token };
          return u;
        });
      },
      onSources: (sources) => {
        setMessages(prev => {
          const u = [...prev];
          const last = u[u.length - 1];
          if (last?.role === "assistant") u[u.length - 1] = { ...last, sources };
          return u;
        });
      },
      onDone: () => {
        setMessages(prev => {
          const u = [...prev];
          const last = u[u.length - 1];
          if (last?.role === "assistant") u[u.length - 1] = { ...last, streaming: false };
          return u;
        });
        setStreaming(false);
        esRef.current = null;
      },
      onError: (err) => {
        setMessages(prev => {
          const u = [...prev];
          const last = u[u.length - 1];
          if (last?.role === "assistant")
            u[u.length - 1] = { ...last, text: `Error: ${err}`, streaming: false, error: true };
          return u;
        });
        setStreaming(false);
      },
    });
  };

  const stopStream = () => {
    esRef.current?.close();
    setStreaming(false);
    setMessages(prev => {
      const u = [...prev];
      const last = u[u.length - 1];
      if (last?.streaming) u[u.length - 1] = { ...last, streaming: false };
      return u;
    });
  };

  return (
    <div className="flex flex-col h-full">

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">

          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center pt-20 gap-3 fade-in">
              <div className="w-10 h-10 rounded-full border border-zinc-800 flex items-center justify-center">
                <Bot size={18} className="text-emerald-500" />
              </div>
              <p className="text-zinc-600 text-sm font-mono text-center">
                Ask anything about the loaded file
              </p>
              <p className="text-zinc-700 text-xs font-mono text-center">
                Retrieval · Generation · Sources
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className="fade-in">
              {msg.role === "user" ? (
                <div className="flex justify-end gap-2.5 items-start">
                  <div className="max-w-[72%] bg-zinc-800 border border-zinc-700 rounded-2xl rounded-tr-sm px-4 py-3">
                    <p className="text-sm text-zinc-100 leading-relaxed">{msg.text}</p>
                  </div>
                  <div className="w-7 h-7 rounded-full bg-zinc-800 border border-zinc-700
                                  flex items-center justify-center shrink-0 mt-0.5">
                    <User size={13} className="text-zinc-400" />
                  </div>
                </div>
              ) : (
                <div className="flex justify-start gap-2.5 items-start">
                  <div className="w-7 h-7 rounded-full bg-zinc-900 border border-emerald-900
                                  flex items-center justify-center shrink-0 mt-0.5">
                    <Bot size={13} className="text-emerald-500" />
                  </div>
                  <div className="max-w-[85%] flex flex-col gap-2">
                    <div className={`
                      rounded-2xl rounded-tl-sm px-4 py-3 text-sm leading-relaxed
                      ${msg.error
                        ? "bg-red-950/40 border border-red-800 text-red-300"
                        : "bg-zinc-900 border border-zinc-800 text-zinc-200"}
                      ${msg.streaming ? "streaming-cursor" : ""}
                    `}>
                      <pre className="whitespace-pre-wrap font-body text-sm leading-relaxed">
                        {msg.text || (msg.streaming ? "" : "…")}
                      </pre>
                    </div>
                    {!msg.streaming && !msg.error && (
                      <SourceCards sources={msg.sources} />
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar */}
      <div className="shrink-0 px-4 pb-6 pt-2">
        <div className="max-w-3xl mx-auto">
          <div className="flex gap-2 bg-zinc-900 border border-zinc-700 rounded-2xl px-4 py-2.5
                          focus-within:border-emerald-800 transition-colors">
            <input
              type="text" value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendQuery()}
              placeholder="Ask a question about the loaded file…"
              disabled={streaming}
              className="flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-600
                         focus:outline-none font-body disabled:opacity-50 min-w-0"
            />
            {streaming ? (
              <button onClick={stopStream}
                className="p-1.5 rounded-lg text-red-400 hover:text-red-300
                           hover:bg-red-950/40 transition-colors shrink-0"
                title="Stop">
                <StopCircle size={17} />
              </button>
            ) : (
              <button onClick={sendQuery} disabled={!input.trim()}
                className="p-1.5 rounded-lg text-emerald-500 hover:text-emerald-400
                           hover:bg-emerald-950/40 transition-colors
                           disabled:opacity-30 disabled:cursor-not-allowed shrink-0">
                <Send size={17} />
              </button>
            )}
          </div>
          <p className="text-center text-[11px] text-zinc-700 font-mono mt-2">
            Phi-3 · FAISS · CLIP · Whisper · Qwen2-VL — fully offline
          </p>
        </div>
      </div>
    </div>
  );
}