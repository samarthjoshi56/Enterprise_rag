import { useState } from "react";
import {
  generateSQL,
  executeSQL,
  getSchema,
  type Text2SQLGenResponse,
  type Text2SQLExecResponse,
  type SchemaResponse,
} from "../api/client";
import {
  Database,
  Play,
  CheckCircle2,
  XCircle,
  Shield,
  Table,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

function SchemaViewer({ schema }: { schema: SchemaResponse }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="glass" style={{ overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "14px 18px",
          cursor: "pointer",
        }}
        onClick={() => setOpen((o) => !o)}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 14 }}>
          <Table size={15} color="var(--accent-primary)" />
          Available Tables ({schema.allowed_tables.length})
        </div>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>
      {open && (
        <div
          style={{
            borderTop: "1px solid var(--border)",
            padding: "12px 18px",
            display: "flex",
            gap: 8,
            flexWrap: "wrap",
          }}
        >
          {schema.allowed_tables.map((t) => (
            <span key={t} className="badge badge-info mono">{t}</span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Text2SQL() {
  const [question, setQuestion] = useState("");
  const [generated, setGenerated] = useState<Text2SQLGenResponse | null>(null);
  const [executed, setExecuted] = useState<Text2SQLExecResponse | null>(null);
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [generating, setGenerating] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(false);

  async function loadSchema() {
    setSchemaLoading(true);
    try {
      setSchema(await getSchema());
    } finally {
      setSchemaLoading(false);
    }
  }

  async function generate() {
    if (!question.trim()) return;
    setError(null);
    setGenerated(null);
    setExecuted(null);
    setGenerating(true);
    try {
      const res = await generateSQL(question);
      setGenerated(res);
      if (!schema) loadSchema();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setGenerating(false);
    }
  }

  async function approve(ok: boolean) {
    if (!generated?.query_id) return;
    setExecuting(true);
    setError(null);
    try {
      const res = await executeSQL(generated.query_id, ok);
      setExecuted(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setExecuting(false);
    }
  }

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.5px" }}>
            <span className="gradient-text">Text2SQL</span>
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
            Natural language → SQL · Human-in-the-loop approval · SELECT-only execution
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={loadSchema} disabled={schemaLoading}>
          <Database size={13} />
          {schemaLoading ? "Loading…" : "View Schema"}
        </button>
      </div>

      {schema && <SchemaViewer schema={schema} />}

      {/* Input */}
      <div className="glass" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ fontWeight: 600, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
          <Database size={15} color="var(--accent-primary)" />
          Natural Language Question
        </div>
        <textarea
          className="input"
          style={{ minHeight: 80, resize: "vertical", fontFamily: "Inter, sans-serif" }}
          placeholder='e.g. "What are the top 5 documents by chunk count?"'
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button className="btn btn-primary" onClick={generate} disabled={generating || !question.trim()}>
            {generating ? <span className="animate-spin" style={{ display: "inline-block" }}>⟳</span> : <Play size={14} />}
            {generating ? "Generating…" : "Generate SQL"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, color: "#ef4444", fontSize: 13 }}>
          {error}
        </div>
      )}

      {/* Generated SQL + approval */}
      {generated && (
        <div className="glass animate-fade-in" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontWeight: 600, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
              <Shield size={15} color={generated.is_safe ? "var(--accent-success)" : "var(--accent-danger)"} />
              Generated SQL
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <span className={`badge ${generated.is_safe ? "badge-success" : "badge-danger"}`}>
                {generated.is_safe ? "Safe" : "Unsafe"}
              </span>
              <span className={`badge ${generated.status === "pending_approval" ? "badge-warning" : "badge-info"}`}>
                {generated.status}
              </span>
            </div>
          </div>

          <div className="code-block" style={{ color: "#79c0ff" }}>
            {generated.sanitized_sql ?? generated.generated_sql}
          </div>

          {generated.explanation && (
            <div style={{ padding: "10px 14px", background: "var(--bg-elevated)", borderRadius: 8, fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
              <strong style={{ color: "var(--text-primary)" }}>Explanation: </strong>{generated.explanation}
            </div>
          )}

          {generated.is_safe && !executed && (
            <div style={{ display: "flex", gap: 10, paddingTop: 4 }}>
              <div style={{ flex: 1, padding: "12px 16px", background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.25)", borderRadius: 10, fontSize: 13, color: "#f59e0b" }}>
                ⚠️ Review the SQL above before approving execution.
              </div>
              <button
                className="btn btn-success"
                onClick={() => approve(true)}
                disabled={executing}
              >
                <CheckCircle2 size={14} />
                {executing ? "Executing…" : "Approve & Run"}
              </button>
              <button
                className="btn btn-danger"
                onClick={() => approve(false)}
                disabled={executing}
              >
                <XCircle size={14} />
                Reject
              </button>
            </div>
          )}

          {!generated.is_safe && generated.error && (
            <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, color: "#ef4444", fontSize: 13 }}>
              🚫 {generated.error}
            </div>
          )}
        </div>
      )}

      {/* Execution results */}
      {executed && (
        <div className="glass animate-fade-in" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontWeight: 600, fontSize: 14, display: "flex", alignItems: "center", gap: 8 }}>
              <CheckCircle2 size={15} color="var(--accent-success)" />
              Query Results
            </div>
            <div style={{ display: "flex", gap: 10, fontSize: 12, color: "var(--text-muted)", alignItems: "center" }}>
              <span>{executed.row_count} rows</span>
              <span>·</span>
              <span>{executed.execution_time_ms.toFixed(1)}ms</span>
              <span className={`badge ${executed.status === "executed" ? "badge-success" : "badge-danger"}`}>
                {executed.status}
              </span>
            </div>
          </div>

          {executed.status === "rejected" ? (
            <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, color: "#ef4444", fontSize: 13 }}>
              Query was rejected.
            </div>
          ) : executed.columns.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>No results returned.</div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="data-table">
                <thead>
                  <tr>
                    {executed.columns.map((c) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {executed.rows.map((row, i) => (
                    <tr key={i}>
                      {executed.columns.map((c) => (
                        <td key={c} style={{ maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {String(row[c] ?? "—")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Quick examples */}
      <div className="glass" style={{ padding: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: "var(--text-secondary)" }}>Quick examples</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {[
            "List all documents with their chunk counts",
            "What are the top 5 largest documents by file size?",
            "How many documents were ingested in the last 7 days?",
            "Show documents grouped by file type",
          ].map((q) => (
            <button
              key={q}
              className="btn btn-secondary btn-sm"
              onClick={() => setQuestion(q)}
              style={{ fontSize: 12 }}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
