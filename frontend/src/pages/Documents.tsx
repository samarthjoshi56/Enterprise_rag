import { useCallback, useEffect, useRef, useState } from "react";
import {
  listDocuments,
  uploadDocument,
  getDocument,
  type DocumentMeta,
  type DocumentDetail,
} from "../api/client";
import {
  Upload,
  ChevronRight,
  FileText,
  X,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
  Layers,
} from "lucide-react";

function StatusBadge({ status }: { status: string }) {
  if (status === "completed") return <span className="badge badge-success"><CheckCircle2 size={10} />Completed</span>;
  if (status === "failed") return <span className="badge badge-danger"><XCircle size={10} />Failed</span>;
  return <span className="badge badge-warning"><Clock size={10} />Processing</span>;
}

function DetailDrawer({ doc, onClose }: { doc: DocumentDetail; onClose: () => void }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        display: "flex",
      }}
    >
      <div
        style={{ flex: 1, background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}
        onClick={onClose}
      />
      <div
        className="animate-slide-in"
        style={{
          width: 460,
          background: "var(--bg-surface)",
          borderLeft: "1px solid var(--border)",
          padding: 28,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontWeight: 700, fontSize: 16 }}>Document Detail</h2>
          <button className="btn btn-secondary btn-sm" onClick={onClose}><X size={14} /></button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[
            ["Filename", doc.filename],
            ["Type", doc.file_type.toUpperCase()],
            ["Size", `${(doc.file_size_bytes / 1024).toFixed(1)} KB`],
            ["Pages", doc.total_pages ?? "N/A"],
            ["Chunks", doc.total_chunks],
            ["Status", <StatusBadge key="s" status={doc.status} />],
            ["Ingested", doc.created_at ? new Date(doc.created_at).toLocaleString() : "—"],
          ].map(([label, val]) => (
            <div
              key={String(label)}
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "8px 12px",
                background: "var(--bg-elevated)",
                borderRadius: 8,
                fontSize: 13,
              }}
            >
              <span style={{ color: "var(--text-muted)" }}>{label}</span>
              <span style={{ fontWeight: 500 }}>{val}</span>
            </div>
          ))}
        </div>
        {doc.error_message && (
          <div style={{ padding: 12, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 8, fontSize: 13, color: "#ef4444" }}>
            {doc.error_message}
          </div>
        )}
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: "var(--text-secondary)" }}>
            Chunks ({doc.chunks.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 320, overflowY: "auto" }}>
            {doc.chunks.map((c) => (
              <div
                key={c.chunk_id}
                style={{
                  padding: "8px 12px",
                  background: "var(--bg-elevated)",
                  borderRadius: 8,
                  border: "1px solid var(--border)",
                  fontSize: 12,
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 8,
                }}
              >
                <span style={{ color: "var(--text-muted)" }}>#{c.chunk_index}</span>
                <span>Page {c.page_number ?? "—"}</span>
                <span style={{ color: "var(--text-secondary)" }}>{c.character_count} chars</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Documents() {
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [detail, setDetail] = useState<DocumentDetail | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function load() {
    setLoading(true);
    try {
      setDocs(await listDocuments());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const file = files[0];
    setError(null);
    setUploading(true);
    setUploadProgress(10);
    try {
      const timer = setInterval(() => setUploadProgress((p) => Math.min(p + 15, 85)), 600);
      await uploadDocument(file);
      clearInterval(timer);
      setUploadProgress(100);
      setTimeout(() => setUploadProgress(0), 800);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  }, []);

  async function openDetail(id: string) {
    const d = await getDocument(id);
    setDetail(d);
  }

  return (
    <div className="page" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.5px" }}>
            <span className="gradient-text">Documents</span>
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: 14, marginTop: 4 }}>
            Upload and manage your knowledge base
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={load}>
          <RefreshCw size={13} />
          Refresh
        </button>
      </div>

      {/* Upload zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !uploading && fileRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? "var(--accent-primary)" : "var(--border)"}`,
          borderRadius: 14,
          padding: "36px 24px",
          textAlign: "center",
          cursor: uploading ? "not-allowed" : "pointer",
          background: dragging ? "rgba(99,102,241,0.06)" : "var(--bg-card)",
          transition: "all 0.2s",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 10,
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt"
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div
          style={{
            width: 52,
            height: 52,
            borderRadius: 14,
            background: "rgba(99,102,241,0.15)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--accent-primary)",
          }}
        >
          <Upload size={22} />
        </div>
        {uploading ? (
          <>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Uploading & ingesting...</div>
            <div
              style={{
                width: 240,
                height: 6,
                background: "var(--bg-elevated)",
                borderRadius: 999,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${uploadProgress}%`,
                  height: "100%",
                  background: "linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))",
                  transition: "width 0.3s",
                  borderRadius: 999,
                }}
              />
            </div>
          </>
        ) : (
          <>
            <div style={{ fontWeight: 600, fontSize: 14 }}>Drop PDF or TXT here, or click to browse</div>
            <div style={{ color: "var(--text-muted)", fontSize: 12 }}>
              Supported: .pdf, .txt · Document will be chunked and embedded automatically
            </div>
          </>
        )}
      </div>

      {error && (
        <div
          style={{
            padding: "12px 16px",
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.3)",
            borderRadius: 10,
            color: "#ef4444",
            fontSize: 13,
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          {error}
          <button onClick={() => setError(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "#ef4444" }}>
            <X size={14} />
          </button>
        </div>
      )}

      {/* Document list */}
      <div className="glass" style={{ overflow: "hidden" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Type</th>
              <th>Size</th>
              <th>Pages</th>
              <th>Chunks</th>
              <th>Status</th>
              <th>Ingested</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              [1, 2, 3].map((i) => (
                <tr key={i}>
                  {Array(8).fill(0).map((_, j) => (
                    <td key={j}><div className="skeleton" style={{ height: 16, borderRadius: 4 }} /></td>
                  ))}
                </tr>
              ))
            ) : docs.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: "center", padding: "40px", color: "var(--text-muted)" }}>
                  No documents yet. Upload one above.
                </td>
              </tr>
            ) : (
              docs.map((d) => (
                <tr key={d.id}>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <FileText size={13} color="var(--accent-primary)" />
                      <span style={{ fontWeight: 500, maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {d.filename}
                      </span>
                    </div>
                  </td>
                  <td><span className="badge badge-info">{d.file_type.toUpperCase()}</span></td>
                  <td style={{ color: "var(--text-secondary)", fontSize: 12 }}>
                    {(d.file_size_bytes / 1024).toFixed(1)} KB
                  </td>
                  <td style={{ color: "var(--text-secondary)" }}>{d.total_pages ?? "—"}</td>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      <Layers size={11} color="var(--accent-secondary)" />
                      {d.total_chunks}
                    </div>
                  </td>
                  <td><StatusBadge status={d.status} /></td>
                  <td style={{ color: "var(--text-muted)", fontSize: 12 }}>
                    {d.created_at ? new Date(d.created_at).toLocaleDateString() : "—"}
                  </td>
                  <td>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => openDetail(d.id)}
                    >
                      <ChevronRight size={13} />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {detail && <DetailDrawer doc={detail} onClose={() => setDetail(null)} />}
    </div>
  );
}
