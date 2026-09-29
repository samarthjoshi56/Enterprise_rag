import { useEffect, useRef, useState } from "react";
import { chatQuery, type ChatSource } from "../api/client";
import { Send, Bot, User, FileText, ChevronDown, ChevronUp, Zap, Sparkles } from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  latency?: number;
  cached?: boolean;
  mode?: string;
  auditTrail?: string[];
}

function SourceCard({ item }: { item: ChatSource }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      style={{
        background: "var(--bg-elevated)",
        border: "1px solid var(--border)",
        borderRadius: 10,
        overflow: "hidden",
        fontSize: 12,
      }}
    >
      <div
        style={{
          padding: "8px 12px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          cursor: "pointer",
          gap: 8,
        }}
        onClick={() => setOpen((o) => !o)}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 0 }}>
          <FileText size={11} color="var(--accent-primary)" style={{ flexShrink: 0 }} />
          <span style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {item.filename}
          </span>
          {item.page_number != null && (
            <span style={{ color: "var(--text-muted)", flexShrink: 0 }}>p.{item.page_number}</span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
          <span className="badge badge-purple">{item.score.toFixed(3)}</span>
          {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </div>
      </div>
      {open && (
        <div
          style={{
            padding: "0 12px 10px",
            color: "var(--text-secondary)",
            lineHeight: 1.6,
            borderTop: "1px solid var(--border)",
            paddingTop: 10,
          }}
        >
          {item.text}
        </div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: 4, padding: "10px 14px" }}>
      {[0, 1, 2].map((i) => (
        <span key={i} className="typing-dot" style={{ animationDelay: `${i * 0.16}s` }} />
      ))}
    </div>
  );
}

export default function Chat() {
  const [useAdvancedRag, setUseAdvancedRag] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi! I'm your Enterprise RAG assistant. Ask me anything about your documents, and I'll retrieve the grounded context and answer with citations.",
    },
  ]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  async function send() {
    const q = input.trim();
    if (!q || typing) return;
    setInput("");

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: q };
    setMessages((m) => [...m, userMsg]);
    setTyping(true);

    try {
      const res = await chatQuery({ query: q, top_n: 5, use_advanced_rag: useAdvancedRag });

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: res.answer,
        sources: res.sources,
        latency: res.latency_ms,
        cached: res.cached,
        mode: res.mode,
        auditTrail: res.audit_trail,
      };
      setMessages((m) => [...m, assistantMsg]);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Error connecting to server";
      setMessages((m) => [
        ...m,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: `⚠️ ${errMsg}`,
        },
      ]);
    } finally {
      setTyping(false);
    }
  }

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 48px)", gap: 0 }}>
      {/* Header */}
      <div style={{ marginBottom: 20, display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.5px" }}>
            <span className="gradient-text">RAG Chat</span>
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
            {useAdvancedRag
              ? "Advanced RAG · HyDE query expansion · CRAG grading · Self-RAG reflection"
              : "Hybrid retrieval (Vector + FTS) · Cross-encoder neural reranking"}
          </p>
        </div>

        <button
          onClick={() => setUseAdvancedRag((v) => !v)}
          className={`btn ${useAdvancedRag ? "btn-primary" : "btn-secondary"}`}
          style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}
        >
          <Sparkles size={14} />
          {useAdvancedRag ? "Advanced RAG: ON" : "Advanced RAG: OFF"}
        </button>
      </div>


      {/* Messages */}
      <div
        className="glass"
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 20,
          display: "flex",
          flexDirection: "column",
          gap: 20,
          marginBottom: 16,
        }}
      >
        {messages.map((msg) => (
          <div
            key={msg.id}
            className="animate-fade-in"
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 10,
              alignItems: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div style={{ display: "flex", gap: 10, alignItems: "flex-end", flexDirection: msg.role === "user" ? "row-reverse" : "row" }}>
              <div
                style={{
                  width: 30,
                  height: 30,
                  borderRadius: "50%",
                  background:
                    msg.role === "user"
                      ? "linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))"
                      : "var(--bg-elevated)",
                  border: "1px solid var(--border)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                {msg.role === "user" ? <User size={14} color="white" /> : <Bot size={14} color="var(--accent-primary)" />}
              </div>
              <div className={`chat-bubble chat-bubble-${msg.role === "user" ? "user" : "assistant"}`}>
                {msg.content.split("\n").map((line, i) => {
                  const bold = line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
                  return (
                    <p key={i} style={{ margin: i > 0 ? "6px 0 0" : 0 }} dangerouslySetInnerHTML={{ __html: bold }} />
                  );
                })}
              </div>
            </div>

            {msg.role === "assistant" && (msg.latency !== undefined || msg.cached !== undefined) && (
              <div style={{ display: "flex", gap: 8, paddingLeft: 40, alignItems: "center" }}>
                {msg.cached && (
                  <span className="badge badge-info"><Zap size={9} />Cached</span>
                )}
                {msg.latency !== undefined && (
                  <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{msg.latency.toFixed(0)}ms</span>
                )}
              </div>
            )}

            {msg.sources && msg.sources.length > 0 && (
              <div style={{ paddingLeft: 40, width: "100%", maxWidth: 640 }}>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" }}>
                  Retrieved Sources ({msg.sources.length})
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {msg.sources.slice(0, 5).map((s, idx) => (
                    <SourceCard key={`${s.filename}-${s.chunk_index}-${idx}`} item={s} />
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}

        {typing && (
          <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
            <div
              style={{
                width: 30,
                height: 30,
                borderRadius: "50%",
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Bot size={14} color="var(--accent-primary)" />
            </div>
            <div className="chat-bubble chat-bubble-assistant">
              <TypingIndicator />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div
        className="glass"
        style={{ padding: "12px 16px", display: "flex", gap: 10, alignItems: "center" }}
      >
        <input
          className="input"
          placeholder="Ask anything about your documents…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
          style={{ flex: 1 }}
          disabled={typing}
        />
        <button
          className="btn btn-primary"
          onClick={send}
          disabled={typing || !input.trim()}
        >
          <Send size={15} />
        </button>
      </div>
    </div>
  );
}
