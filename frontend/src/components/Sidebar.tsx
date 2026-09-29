import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  Search,
  Database,
  BarChart3,
  Settings,
  Cpu,
} from "lucide-react";

const nav = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/documents", icon: FileText, label: "Documents" },
  { to: "/chat", icon: MessageSquare, label: "RAG Chat" },
  { to: "/search", icon: Search, label: "Search" },
  { to: "/text2sql", icon: Database, label: "Text2SQL" },
  { to: "/evaluation", icon: BarChart3, label: "Evaluation" },
  { to: "/settings", icon: Settings, label: "Settings" },
];

export default function Sidebar() {
  return (
    <aside
      style={{
        width: 220,
        minWidth: 220,
        background: "var(--bg-surface)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        padding: "20px 12px",
        gap: 4,
        height: "100vh",
        position: "sticky",
        top: 0,
      }}
    >
      {/* Logo */}
      <div style={{ padding: "8px 8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <div
          style={{
            width: 34,
            height: 34,
            borderRadius: 10,
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 4px 16px rgba(99,102,241,0.4)",
          }}
        >
          <Cpu size={18} color="white" />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: "var(--text-primary)" }}>Enterprise</div>
          <div style={{ fontWeight: 400, fontSize: 11, color: "var(--text-muted)" }}>RAG Platform</div>
        </div>
      </div>

      {/* Nav */}
      {nav.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
        >
          <Icon size={16} />
          {label}
        </NavLink>
      ))}

      <div style={{ flex: 1 }} />
      <div
        style={{
          padding: "12px",
          borderRadius: 10,
          background: "var(--bg-elevated)",
          border: "1px solid var(--border)",
          fontSize: 11,
          color: "var(--text-muted)",
          lineHeight: 1.6,
        }}
      >
        <div style={{ fontWeight: 600, color: "var(--text-secondary)", marginBottom: 2 }}>Backend</div>
        <div>FastAPI · 127.0.0.1:8000</div>
      </div>
    </aside>
  );
}
