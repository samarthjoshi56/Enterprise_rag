const API_BASE = import.meta.env.VITE_API_URL || "";
const BASE = `${API_BASE}/api/v1`;

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

// ─── Health ────────────────────────────────────────────────────────────────
export interface HealthResponse {
  status: string;
  app_name: string;
  environment: string;
  timestamp: string;
  services: {
    postgres: { status: string; latency_ms?: number };
    qdrant: { status: string; version?: string; collections?: number };
    redis: { status: string; latency_ms?: number };
  };
  frameworks: Record<string, { status: string; version?: string }>;
}
export const fetchHealth = () => request<HealthResponse>("/health");

// ─── Documents ────────────────────────────────────────────────────────────
export interface DocumentMeta {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  total_pages: number | null;
  total_chunks: number;
  status: string;
  created_at: string | null;
}
export interface DocumentDetail extends DocumentMeta {
  error_message: string | null;
  chunks: Array<{
    chunk_id: string;
    chunk_index: number;
    page_number: number | null;
    qdrant_point_id: string | null;
    character_count: number;
  }>;
}

export const listDocuments = () => request<DocumentMeta[]>("/documents");
export const getDocument = (id: string) => request<DocumentDetail>(`/documents/${id}`);

export const uploadDocument = async (file: File): Promise<DocumentMeta> => {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/documents/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
};

// ─── Search ────────────────────────────────────────────────────────────────
export interface SearchRequest {
  query: string;
  vector_top_k?: number;
  keyword_top_k?: number;
  alpha?: number;
  top_n?: number;
  bypass_cache?: boolean;
}
export interface SearchResultItem {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number | null;
  chunk_index: number;
  text: string;
  vector_score: number;
  keyword_score: number;
  hybrid_score: number;
  rerank_score: number | null;
}
export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  total_candidates: number;
  vector_hits: number;
  keyword_hits: number;
  returned: number;
  latency_ms: number;
  cached: boolean;
  cache_key: string | null;
}
export const hybridSearch = (body: SearchRequest) =>
  request<SearchResponse>("/search", { method: "POST", body: JSON.stringify(body) });

// ─── Text2SQL ──────────────────────────────────────────────────────────────
export interface Text2SQLGenResponse {
  query_id: string | null;
  question: string;
  generated_sql: string;
  sanitized_sql: string | null;
  explanation: string;
  is_safe: boolean;
  error: string | null;
  status: string;
  allowed_tables: string[];
}
export interface Text2SQLExecResponse {
  query_id: string;
  status: string;
  message: string;
  sql: string;
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  execution_time_ms: number;
}
export interface SchemaResponse {
  allowed_tables: string[];
  schema: Record<string, unknown>;
}

export const generateSQL = (question: string) =>
  request<Text2SQLGenResponse>("/text2sql/generate", {
    method: "POST",
    body: JSON.stringify({ question }),
  });

export const executeSQL = (query_id: string, approved: boolean) =>
  request<Text2SQLExecResponse>("/text2sql/execute", {
    method: "POST",
    body: JSON.stringify({ query_id, approved }),
  });

export const getSchema = () => request<SchemaResponse>("/text2sql/schema");

// ─── Cache ──────────────────────────────────────────────────────────────────
export interface CacheStats {
  enabled: boolean;
  connected: boolean;
  cached_rag_keys: number;
  keyspace_hits?: number;
  keyspace_misses?: number;
  used_memory_human?: string;
  hit_rate?: number;
}
export const getCacheStats = () => request<CacheStats>("/cache/stats");
export const clearCache = (prefix?: string) =>
  request<{ cleared_keys_count: number; message: string }>("/cache/clear", {
    method: "POST",
    body: JSON.stringify({ prefix: prefix ?? null }),
  });

// ─── Evaluation ────────────────────────────────────────────────────────────
export interface EvalDataset {
  total_samples: number;
  samples: Array<{
    id: string;
    question: string;
    ground_truth_keywords: string[];
    expected_answer: string;
    target_service: string;
  }>;
}
export interface EvalReport {
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  hit_rate: number;
  k: number;
  total_queries: number;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
}
export interface ComparisonReport {
  configurations: Record<string, EvalReport & { name: string }>;
}

export const getEvalDataset = () => request<EvalDataset>("/evaluation/dataset");
export const runEvaluation = (k = 5) =>
  request<EvalReport>("/evaluation/run", { method: "POST", body: JSON.stringify({ k }) });
export const compareConfigs = (k = 5) =>
  request<ComparisonReport>("/evaluation/compare", { method: "POST", body: JSON.stringify({ k }) });

// ─── Chat ──────────────────────────────────────────────────────────────────
export interface ChatSource {
  filename: string;
  page_number: number | null;
  chunk_index: number;
  score: number;
  text: string;
}

export interface ChatResponse {
  query: string;
  answer: string;
  sources: ChatSource[];
  latency_ms: number;
  cached: boolean;
  mode: string;
  audit_trail?: string[];
}

export const chatQuery = (body: { query: string; top_n?: number; use_advanced_rag?: boolean }) =>
  request<ChatResponse>("/chat", { method: "POST", body: JSON.stringify(body) });

