import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

function App() {
  const [locale, setLocale] = useState<"zh-CN" | "en">("zh-CN");
  const text = locale === "zh-CN" ? {
    projects: "工作区", threads: "线程", newThread: "新建任务", placeholder: "告诉 Justin 你想完成什么…",
    plan: "计划", terminal: "终端", tools: "工具", changes: "变更", approvals: "审批",
    welcome: "Codex 内核已就绪", description: "选择工作区后开始一个任务。代理执行、审批和会话由 Codex App Server 管理。",
  } : {
    projects: "Workspaces", threads: "Threads", newThread: "New task", placeholder: "Tell Justin what you want to accomplish…",
    plan: "Plan", terminal: "Terminal", tools: "Tools", changes: "Changes", approvals: "Approvals",
    welcome: "Codex kernel is ready", description: "Choose a workspace to start a task. Codex App Server owns execution, approvals, and conversations.",
  };
  return <main>
    <aside><header>Justin Agent <button onClick={() => setLocale(locale === "zh-CN" ? "en" : "zh-CN")}>{locale === "zh-CN" ? "EN" : "中"}</button></header>
      <h2>{text.projects}</h2><button className="workspace">+ F:\\Codex Project</button><h2>{text.threads}</h2><button className="thread active">{text.newThread}</button></aside>
    <section className="conversation"><div className="empty"><span>✦</span><h1>{text.welcome}</h1><p>{text.description}</p></div>
      <form onSubmit={(event) => event.preventDefault()}><textarea aria-label={text.placeholder} placeholder={text.placeholder}/><button>↑</button></form></section>
    <aside className="inspector"><h2>{text.plan}</h2><p className="muted">—</p><nav>{[text.terminal, text.tools, text.changes, text.approvals].map((item) => <button key={item}>{item}</button>)}</nav></aside>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
