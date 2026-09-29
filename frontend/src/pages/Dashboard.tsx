import { useEffect, useState } from "react";
import {
  fetchHealth,
  listDocuments,
  type HealthResponse,
  type DocumentMeta,
} from "../api/client";
import StatCard from "../components/StatCard";
import {
  FileText,
  Layers,
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
} from "lucide-react";

function ServiceDot({ status }: { status: string }) {
  const ok = status === "connected" || status === "healthy";
  return (
    <span
      style={{
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: ok ? "var(--accent-success)" : "var(--accent-danger)",
        display: "inline-block",
        boxShadow: ok ? "0 0 6px #10b981" : "0 0 6px #ef4444",
      }}
    />
  );
}

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  async function load() {
    setRefreshing(true);
    try {
      const [h, d] = await Promise.all([fetchHealth(), listDocuments()]);
      setHealth(h);
      setDocs(d);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => { load(); }, []);

  const totalChunks = docs.reduce((s, d) => s + (d.total_chunks ?? 0), 0);
  const completedDocs = docs.filter((d) => d.status === "completed").length;

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.5px" }}>
            <span className="gradient-text">Dashboard</span>
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
            System overview and health status
          </p>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          onClick={load}
          disabled={refreshing}
        >
          <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {/* Stat cards */}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <StatCard
          label="Documents"
          value={loading ? "—" : docs.length}
          sub={`${completedDocs} processed`}
          icon={<FileText size={16} />}
          accent="#6366f1"
          loading={loading}
        />
        <StatCard
          label="Total Chunks"
          value={loading ? "—" : totalChunks.toLocaleString()}
          sub="stored in Qdrant"
          icon={<Layers size={16} />}
          accent="#8b5cf6"
          loading={loading}
        />
        <StatCard
          label="System Status"
          value={loading ? "—" : health?.status?.toUpperCase() ?? "UNKNOWN"}
          sub={health?.environment ?? ""}
          icon={<Activity size={16} />}
          accent={health?.status === "healthy" ? "#10b981" : "#ef4444"}
          loading={loading}
        />
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {/* Services */}
        <div className="glass" style={{ flex: 1, minWidth: 280, padding: 24 }}>
          <div style={{ fontWeight: 700, marginBottom: 18, fontSize: 15 }}>Service Health</div>
          {loading ? (
            [1, 2, 3].map((i) => (
              <div key={i} className="skeleton" style={{ height: 20, marginBottom: 12, borderRadius: 6 }} />
            ))
          ) : health ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {Object.entries(health.services).map(([svc, info]) => (
                <div
                  key={svc}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "10px 14px",
                    background: "var(--bg-elevated)",
                    borderRadius: 10,
                    border: "1px solid var(--border)",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <ServiceDot status={info.status} />
                    <span style={{ fontWeight: 600, textTransform: "capitalize", fontSize: 14 }}>{svc}</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span className={`badge ${info.status === "connected" ? "badge-success" : "badge-danger"}`}>
                      {info.status}
                    </span>
                    {(info as { latency_ms?: number }).latency_ms !== undefined && (
                      <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                        {(info as { latency_ms: number }).latency_ms.toFixed(1)}ms
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Could not load health data.</div>
          )}
        </div>

        {/* Recent documents */}
        <div className="glass" style={{ flex: 2, minWidth: 320, padding: 24 }}>
          <div style={{ fontWeight: 700, marginBottom: 18, fontSize: 15 }}>Recent Documents</div>
          {loading ? (
            [1, 2, 3].map((i) => (
              <div key={i} className="skeleton" style={{ height: 44, marginBottom: 10, borderRadius: 8 }} />
            ))
          ) : docs.length === 0 ? (
            <div
              style={{
                textAlign: "center",
                padding: "32px",
                color: "var(--text-muted)",
                fontSize: 13,
              }}
            >
              No documents ingested yet.
              <br />
              Upload some in the Documents tab.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {docs.slice(0, 8).map((d) => (
                <div
                  key={d.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "10px 14px",
                    background: "var(--bg-elevated)",
                    borderRadius: 10,
                    border: "1px solid var(--border)",
                    gap: 12,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
                    <FileText size={14} color="var(--accent-primary)" style={{ flexShrink: 0 }} />
                    <span
                      style={{
                        fontSize: 13,
                        fontWeight: 500,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {d.filename}
                    </span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
                    {d.status === "completed" ? (
                      <CheckCircle2 size={14} color="var(--accent-success)" />
                    ) : d.status === "failed" ? (
                      <XCircle size={14} color="var(--accent-danger)" />
                    ) : (
                      <Clock size={14} color="var(--accent-warning)" />
                    )}
                    <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
                      {d.total_chunks} chunks
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
