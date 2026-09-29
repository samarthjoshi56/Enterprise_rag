import type { ReactNode } from "react";

interface Props {
  label: string;
  value: string | number;
  sub?: string;
  icon: ReactNode;
  accent?: string;
  loading?: boolean;
}

export default function StatCard({ label, value, sub, icon, accent = "#6366f1", loading }: Props) {
  return (
    <div
      className="glass"
      style={{ padding: "20px 24px", display: "flex", flexDirection: "column", gap: 12, flex: 1 }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ color: "var(--text-secondary)", fontSize: 13, fontWeight: 500 }}>{label}</div>
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 10,
            background: `${accent}22`,
            border: `1px solid ${accent}44`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: accent,
          }}
        >
          {icon}
        </div>
      </div>
      {loading ? (
        <div className="skeleton" style={{ height: 32, width: "60%" }} />
      ) : (
        <div style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.5px", color: "var(--text-primary)" }}>
          {value}
        </div>
      )}
      {sub && (
        <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{sub}</div>
      )}
    </div>
  );
}
