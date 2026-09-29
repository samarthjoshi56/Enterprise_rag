import { useEffect, useState } from "react";
import { fetchHealth, getCacheStats, clearCache, type HealthResponse, type CacheStats } from "../api/client";
import { Settings2, Trash2, RefreshCw, Server, Cpu, Zap } from "lucide-react";

function ServiceRow({ name, info }: { name: string; info: { status: string; version?: string; latency_ms?: number; collections?: number } }) {
  const ok = info.status === "connected";
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", background: "var(--bg-elevated)", borderRadius: 10, border: "1px solid var(--border)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: ok ? "var(--accent-success)" : "var(--accent-danger)", display: "inline-block", boxShadow: ok ? "0 0 6px #10b981" : "0 0 6px #ef4444" }} />
        <span style={{ fontWeight: 600, textTransform: "capitalize", fontSize: 14 }}>{name}</span>
      </div>
      <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 12 }}>
        {info.version && <span className="badge badge-info">v{info.version}</span>}
        {info.collections !== undefined && <span style={{ color: "var(--text-muted)" }}>{info.collections} collections</span>}
        {info.latency_ms !== undefined && <span style={{ color: "var(--text-muted)" }}>{info.latency_ms.toFixed(1)}ms</span>}
        <span className={`badge ${ok ? "badge-success" : "badge-danger"}`}>{info.status}</span>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(false);
  const [clearMsg, setClearMsg] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    try {
      const [h, cs] = await Promise.all([fetchHealth(), getCacheStats()]);
      setHealth(h);
      setCacheStats(cs);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleClearCache() {
    setClearing(true);
    setClearMsg(null);
    try {
      const res = await clearCache();
      setClearMsg(res.message);
      load();
    } catch (e) {
      setClearMsg(`Error: ${(e as Error).message}`);
    } finally {
      setClearing(false);
    }
  }

  const models = [
    { label: "Embedding Model", value: "all-MiniLM-L6-v2", sub: "384-dim · sentence-transformers" },
    { label: "Reranker Model", value: "ms-marco-MiniLM-L-6-v2", sub: "Cross-encoder · sentence-transformers" },
    { label: "LLM (Text2SQL)", value: "OpenAI GPT-4o-mini", sub: "Via LangChain" },
  ];

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.5px" }}>
            <span className="gradient-text">Settings</span>
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
            Backend status · Model configuration · Cache management
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={load}>
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {/* Backend status */}
      <div className="glass" style={{ padding: 24 }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}>
          <Server size={16} color="var(--accent-primary)" />
          Backend Services
        </div>
        {loading ? (
          [1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 44, marginBottom: 10, borderRadius: 10 }} />)
        ) : health ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {Object.entries(health.services).map(([name, info]) => (
              <ServiceRow key={name} name={name} info={info as Parameters<typeof ServiceRow>[0]["info"]} />
            ))}
          </div>
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Could not reach backend.</div>
        )}

        {health && (
          <div style={{ marginTop: 16, display: "flex", gap: 12, flexWrap: "wrap" }}>
            {[
              ["App", health.app_name],
              ["Environment", health.environment],
              ["Overall Status", health.status],
              ["Timestamp", new Date(health.timestamp).toLocaleTimeString()],
            ].map(([label, val]) => (
              <div key={String(label)} style={{ padding: "8px 14px", background: "var(--bg-elevated)", borderRadius: 8, fontSize: 12, border: "1px solid var(--border)" }}>
                <span style={{ color: "var(--text-muted)" }}>{label}: </span>
                <span style={{ fontWeight: 600 }}>{val}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Model config */}
      <div className="glass" style={{ padding: 24 }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}>
          <Cpu size={16} color="var(--accent-secondary)" />
          Model Configuration
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {models.map((m) => (
            <div
              key={m.label}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "12px 16px",
                background: "var(--bg-elevated)",
                borderRadius: 10,
                border: "1px solid var(--border)",
              }}
            >
              <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{m.label}</div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{m.value}</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{m.sub}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Frameworks */}
      {health?.frameworks && (
        <div className="glass" style={{ padding: 24 }}>
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}>
            <Settings2 size={16} color="var(--accent-tertiary)" />
            Framework Versions
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {Object.entries(health.frameworks).map(([name, info]) => (
              <div
                key={name}
                style={{
                  padding: "10px 14px",
                  background: "var(--bg-elevated)",
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                  fontSize: 12,
                }}
              >
                <div style={{ fontWeight: 600, textTransform: "capitalize" }}>{name.replace(/_/g, " ")}</div>
                <div style={{ color: "var(--text-muted)", marginTop: 2 }}>
                  {info.status === "available" ? (
                    <span style={{ color: "var(--accent-success)" }}>v{info.version}</span>
                  ) : (
                    <span style={{ color: "var(--accent-danger)" }}>unavailable</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cache */}
      <div className="glass" style={{ padding: 24 }}>
        <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 18, display: "flex", alignItems: "center", gap: 8 }}>
          <Zap size={16} color="var(--accent-warning)" />
          Redis Cache
        </div>
        {cacheStats ? (
          <>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 16 }}>
              {[
                ["Status", cacheStats.connected ? "Connected" : "Disconnected"],
                ["Enabled", cacheStats.enabled ? "Yes" : "No"],
                ["Cached queries", cacheStats.cached_rag_keys],
                ...(cacheStats.used_memory_human ? [["Memory", cacheStats.used_memory_human]] : []),
                ...(cacheStats.hit_rate != null ? [["Hit rate", `${(cacheStats.hit_rate * 100).toFixed(1)}%`]] : []),
                ...(cacheStats.keyspace_hits != null ? [["Cache hits", cacheStats.keyspace_hits]] : []),
                ...(cacheStats.keyspace_misses != null ? [["Cache misses", cacheStats.keyspace_misses]] : []),
              ].map(([label, val]) => (
                <div
                  key={String(label)}
                  style={{
                    padding: "10px 14px",
                    background: "var(--bg-elevated)",
                    borderRadius: 10,
                    border: "1px solid var(--border)",
                    fontSize: 12,
                  }}
                >
                  <div style={{ color: "var(--text-muted)" }}>{label}</div>
                  <div style={{ fontWeight: 700, fontSize: 15, marginTop: 2 }}>{val}</div>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <button className="btn btn-danger btn-sm" onClick={handleClearCache} disabled={clearing}>
                <Trash2 size={13} />
                {clearing ? "Clearing…" : "Clear All Cache"}
              </button>
              {clearMsg && (
                <span style={{ fontSize: 13, color: clearMsg.startsWith("Error") ? "#ef4444" : "var(--accent-success)" }}>
                  {clearMsg}
                </span>
              )}
            </div>
          </>
        ) : loading ? (
          <div className="skeleton" style={{ height: 60, borderRadius: 10 }} />
        ) : (
          <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Cache stats unavailable.</div>
        )}
      </div>
    </div>
  );
}
