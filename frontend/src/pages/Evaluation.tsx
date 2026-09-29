import { useState } from "react";
import {
  runEvaluation,
  compareConfigs,
  getEvalDataset,
  type EvalReport,
  type ComparisonReport,
  type EvalDataset,
} from "../api/client";
import { BarChart3, Play, BookOpen, GitCompare, TrendingUp } from "lucide-react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

function MetricCard({ label, value, description }: { label: string; value: number; description?: string }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? "#10b981" : pct >= 40 ? "#f59e0b" : "#ef4444";
  return (
    <div
      className="glass"
      style={{
        padding: "18px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        flex: 1,
        minWidth: 150,
      }}
    >
      <div style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </div>
      <div style={{ fontSize: 32, fontWeight: 800, color, letterSpacing: "-1px" }}>{pct}%</div>
      <div className="score-bar">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      {description && <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{description}</div>}
    </div>
  );
}

export default function Evaluation() {
  const [report, setReport] = useState<EvalReport | null>(null);
  const [comparison, setComparison] = useState<ComparisonReport | null>(null);
  const [dataset, setDataset] = useState<EvalDataset | null>(null);
  const [running, setRunning] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [loadingDataset, setLoadingDataset] = useState(false);
  const [k, setK] = useState(5);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"run" | "compare" | "dataset">("run");

  async function runEval() {
    setRunning(true);
    setError(null);
    try {
      setReport(await runEvaluation(k));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  async function runCompare() {
    setComparing(true);
    setError(null);
    try {
      setComparison(await compareConfigs(k));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setComparing(false);
    }
  }

  async function loadDataset() {
    setLoadingDataset(true);
    try {
      setDataset(await getEvalDataset());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingDataset(false);
    }
  }

  // Build radar data for single run
  const radarData = report
    ? [
        { metric: "Precision@K", value: report.precision_at_k * 100 },
        { metric: "Recall@K", value: report.recall_at_k * 100 },
        { metric: "MRR", value: report.mrr * 100 },
        { metric: "Hit Rate", value: report.hit_rate * 100 },
      ]
    : [];

  // Build bar data for comparison
  const barData = comparison
    ? Object.entries(comparison.configurations).map(([key, cfg]) => ({
        name: cfg.name ?? key,
        "Precision@K": Math.round(cfg.precision_at_k * 100),
        "Recall@K": Math.round(cfg.recall_at_k * 100),
        MRR: Math.round(cfg.mrr * 100),
        "Hit Rate": Math.round(cfg.hit_rate * 100),
      }))
    : [];

  const tabs = [
    { id: "run" as const, icon: Play, label: "Run Evaluation" },
    { id: "compare" as const, icon: GitCompare, label: "Compare Configs" },
    { id: "dataset" as const, icon: BookOpen, label: "Dataset" },
  ];

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.5px" }}>
          <span className="gradient-text">Evaluation</span>
        </h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
          Retrieval benchmarks · Generation metrics · Configuration comparison
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 6, background: "var(--bg-surface)", padding: 4, borderRadius: 12, border: "1px solid var(--border)", alignSelf: "flex-start" }}>
        {tabs.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "8px 16px",
              borderRadius: 8,
              border: "none",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: 600,
              transition: "all 0.18s",
              background: activeTab === id ? "var(--bg-elevated)" : "transparent",
              color: activeTab === id ? "var(--accent-primary)" : "var(--text-muted)",
              boxShadow: activeTab === id ? "0 2px 8px rgba(0,0,0,0.2)" : "none",
            }}
          >
            <Icon size={13} />
            {label}
          </button>
        ))}
      </div>

      {/* K slider */}
      <div className="glass" style={{ padding: 16, display: "flex", alignItems: "center", gap: 16 }}>
        <label style={{ fontSize: 13, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
          Top-K cutoff:
        </label>
        <input
          type="range"
          min={1}
          max={20}
          step={1}
          value={k}
          onChange={(e) => setK(Number(e.target.value))}
          style={{ flex: 1, accentColor: "var(--accent-primary)" }}
        />
        <span style={{ fontWeight: 700, width: 24, textAlign: "center" }}>{k}</span>

        {activeTab === "run" && (
          <button className="btn btn-primary" onClick={runEval} disabled={running}>
            {running ? <span className="animate-spin" style={{ display: "inline-block" }}>⟳</span> : <Play size={14} />}
            {running ? "Running…" : "Run"}
          </button>
        )}
        {activeTab === "compare" && (
          <button className="btn btn-primary" onClick={runCompare} disabled={comparing}>
            {comparing ? <span className="animate-spin" style={{ display: "inline-block" }}>⟳</span> : <GitCompare size={14} />}
            {comparing ? "Comparing…" : "Compare"}
          </button>
        )}
        {activeTab === "dataset" && (
          <button className="btn btn-secondary" onClick={loadDataset} disabled={loadingDataset}>
            {loadingDataset ? <span className="animate-spin" style={{ display: "inline-block" }}>⟳</span> : <BookOpen size={14} />}
            Load
          </button>
        )}
      </div>

      {error && (
        <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, color: "#ef4444", fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* Run tab */}
      {activeTab === "run" && report && (
        <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <MetricCard label={`Precision@${k}`} value={report.precision_at_k} description="Fraction of retrieved that are relevant" />
            <MetricCard label={`Recall@${k}`} value={report.recall_at_k} description="Fraction of relevant that were retrieved" />
            <MetricCard label="MRR" value={report.mrr} description="Mean Reciprocal Rank" />
            <MetricCard label="Hit Rate" value={report.hit_rate} description="Queries with at least one hit" />
          </div>
          <div className="glass" style={{ padding: 20 }}>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
              <TrendingUp size={15} color="var(--accent-primary)" />
              Radar Overview
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--text-muted)", fontSize: 12 }} />
                <Radar
                  name="Score (%)"
                  dataKey="value"
                  fill="rgba(99,102,241,0.3)"
                  stroke="var(--accent-primary)"
                  strokeWidth={2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="glass" style={{ padding: 16, display: "flex", gap: 16, flexWrap: "wrap", fontSize: 13 }}>
            {[
              ["Queries evaluated", report.total_queries],
              ["K", report.k],
              ...(report.latency_p50_ms ? [["p50 latency", `${report.latency_p50_ms.toFixed(0)}ms`]] : []),
              ...(report.latency_p95_ms ? [["p95 latency", `${report.latency_p95_ms.toFixed(0)}ms`]] : []),
            ].map(([label, val]) => (
              <div key={String(label)} style={{ display: "flex", gap: 6 }}>
                <span style={{ color: "var(--text-muted)" }}>{label}:</span>
                <span style={{ fontWeight: 700 }}>{val}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Compare tab */}
      {activeTab === "compare" && comparison && (
        <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div className="glass" style={{ padding: 20 }}>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
              <BarChart3 size={15} color="var(--accent-primary)" />
              Configuration Comparison (%)
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barData} barCategoryGap="20%">
                <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", borderRadius: 8 }}
                  labelStyle={{ color: "var(--text-primary)", fontWeight: 600 }}
                  itemStyle={{ color: "var(--text-secondary)" }}
                />
                <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-muted)" }} />
                <Bar dataKey="Precision@K" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall@K" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="MRR" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Hit Rate" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="glass" style={{ overflow: "hidden" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Configuration</th>
                  <th>Precision@K</th>
                  <th>Recall@K</th>
                  <th>MRR</th>
                  <th>Hit Rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.configurations).map(([key, cfg]) => (
                  <tr key={key}>
                    <td style={{ fontWeight: 600 }}>{cfg.name ?? key}</td>
                    {(["precision_at_k", "recall_at_k", "mrr", "hit_rate"] as const).map((m) => {
                      const pct = Math.round(cfg[m] * 100);
                      const color = pct >= 70 ? "#10b981" : pct >= 40 ? "#f59e0b" : "#ef4444";
                      return (
                        <td key={m}>
                          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <div className="score-bar" style={{ width: 60 }}>
                              <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
                            </div>
                            <span style={{ fontWeight: 700, color }}>{pct}%</span>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Dataset tab */}
      {activeTab === "dataset" && dataset && (
        <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
            {dataset.total_samples} benchmark samples loaded.
          </div>
          {dataset.samples.map((s) => (
            <div key={s.id} className="glass" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13 }}>{s.question}</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {s.ground_truth_keywords.map((kw) => (
                  <span key={kw} className="badge badge-purple">{kw}</span>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8, fontSize: 11 }}>
                <span className="badge badge-info">{s.target_service}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
