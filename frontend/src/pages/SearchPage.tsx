import { useState } from "react";
import { hybridSearch, type SearchResponse, type SearchResultItem } from "../api/client";
import { Search as SearchIcon, Zap, FileText, ChevronDown, ChevronUp, SlidersHorizontal } from "lucide-react";

function ScoreBar({ value, max = 1 }: { value: number; max?: number }) {
  return (
    <div className="score-bar" style={{ width: 80 }}>
      <div className="score-bar-fill" style={{ width: `${Math.min((value / max) * 100, 100)}%` }} />
    </div>
  );
}

function ResultCard({ item, rank }: { item: SearchResultItem; rank: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="glass"
      style={{ padding: 0, overflow: "hidden" }}
    >
      <div
        style={{
          display: "flex",
          gap: 14,
          padding: "14px 16px",
          cursor: "pointer",
          alignItems: "flex-start",
        }}
        onClick={() => setOpen((o) => !o)}
      >
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: 8,
            background: "linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 700,
            fontSize: 12,
            color: "white",
            flexShrink: 0,
          }}
        >
          {rank}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <FileText size={12} color="var(--accent-primary)" />
            <span style={{ fontWeight: 600, fontSize: 13 }}>{item.filename}</span>
            {item.page_number != null && (
              <span className="badge badge-info">p.{item.page_number}</span>
            )}
            {item.rerank_score != null && (
              <span className="badge badge-purple">rerank: {item.rerank_score.toFixed(3)}</span>
            )}
          </div>
          <p
            style={{
              color: "var(--text-secondary)",
              fontSize: 13,
              lineHeight: 1.6,
              overflow: "hidden",
              display: "-webkit-box",
              WebkitLineClamp: open ? "unset" as unknown as number : 2,
              WebkitBoxOrient: "vertical",
            }}
          >
            {item.text}
          </p>
        </div>
        <div style={{ flexShrink: 0, paddingTop: 2 }}>
          {open ? <ChevronUp size={14} color="var(--text-muted)" /> : <ChevronDown size={14} color="var(--text-muted)" />}
        </div>
      </div>

      {open && (
        <div
          style={{
            borderTop: "1px solid var(--border)",
            padding: "12px 16px",
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 12,
            fontSize: 12,
          }}
        >
          {[
            ["Vector Score", item.vector_score],
            ["Keyword Score", item.keyword_score],
            ["Hybrid Score", item.hybrid_score],
          ].map(([label, val]) => (
            <div key={String(label)}>
              <div style={{ color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <ScoreBar value={Number(val)} />
                <span style={{ fontWeight: 600 }}>{Number(val).toFixed(4)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showOptions, setShowOptions] = useState(false);
  const [alpha, setAlpha] = useState(0.5);
  const [topN, setTopN] = useState(5);
  const [bypassCache, setBypassCache] = useState(false);

  async function search() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await hybridSearch({ query, alpha, top_n: topN, bypass_cache: bypassCache });
      setResponse(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.5px" }}>
          <span className="gradient-text">Hybrid Search</span>
        </h1>
        <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
          Vector + Keyword · RRF fusion · Cross-encoder reranking
        </p>
      </div>

      {/* Search box */}
      <div className="glass" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", gap: 10 }}>
          <div style={{ position: "relative", flex: 1 }}>
            <SearchIcon
              size={16}
              color="var(--text-muted)"
              style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)" }}
            />
            <input
              className="input"
              style={{ paddingLeft: 38 }}
              placeholder="Enter a search query…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && search()}
            />
          </div>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setShowOptions((o) => !o)}
          >
            <SlidersHorizontal size={14} />
          </button>
          <button className="btn btn-primary" onClick={search} disabled={loading || !query.trim()}>
            {loading ? <span className="animate-spin">⟳</span> : <SearchIcon size={15} />}
            Search
          </button>
        </div>

        {showOptions && (
          <div
            className="animate-fade-in"
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 16,
              padding: "12px 4px 0",
              borderTop: "1px solid var(--border)",
            }}
          >
            <div>
              <label style={{ fontSize: 12, color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                Alpha (vector weight): <strong style={{ color: "var(--text-primary)" }}>{alpha}</strong>
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent-primary)" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                <span>Keyword</span><span>Vector</span>
              </div>
            </div>
            <div>
              <label style={{ fontSize: 12, color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                Top-N results: <strong style={{ color: "var(--text-primary)" }}>{topN}</strong>
              </label>
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                style={{ width: "100%", accentColor: "var(--accent-primary)" }}
              />
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, paddingTop: 20 }}>
              <input
                type="checkbox"
                id="bypass-cache"
                checked={bypassCache}
                onChange={(e) => setBypassCache(e.target.checked)}
                style={{ accentColor: "var(--accent-primary)", width: 14, height: 14 }}
              />
              <label htmlFor="bypass-cache" style={{ fontSize: 13, cursor: "pointer" }}>Bypass cache</label>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, color: "#ef4444", fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* Results summary */}
      {response && (
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {[
            ["Results", response.returned],
            ["Candidates", response.total_candidates],
            ["Vector hits", response.vector_hits],
            ["Keyword hits", response.keyword_hits],
            ["Latency", `${response.latency_ms.toFixed(0)}ms`],
          ].map(([label, val]) => (
            <div
              key={String(label)}
              className="glass"
              style={{ padding: "10px 16px", display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}
            >
              <span style={{ color: "var(--text-muted)" }}>{label}:</span>
              <span style={{ fontWeight: 700 }}>{val}</span>
            </div>
          ))}
          {response.cached && (
            <div className="glass" style={{ padding: "10px 16px", display: "flex", gap: 6, alignItems: "center", fontSize: 13 }}>
              <Zap size={13} color="var(--accent-tertiary)" />
              <span style={{ color: "var(--accent-tertiary)", fontWeight: 600 }}>Cached</span>
            </div>
          )}
        </div>
      )}

      {/* Result cards */}
      {loading ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {[1, 2, 3].map((i) => (
            <div key={i} className="skeleton" style={{ height: 80, borderRadius: 14 }} />
          ))}
        </div>
      ) : response?.results.length === 0 ? (
        <div className="glass" style={{ padding: 40, textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
          No results found. Try a different query or upload more documents.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {response?.results.map((r, i) => (
            <ResultCard key={r.chunk_id} item={r} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
