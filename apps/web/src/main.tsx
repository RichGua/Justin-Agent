import { FormEvent, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type ChatMessage = { role: "user" | "assistant"; text: string };
type ThreadSummary = { id: string; name?: string; preview?: string };
function App() {
  const [threadId, setThreadId] = useState<string>();
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState("离线");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const socket = useRef<WebSocket | undefined>(undefined);
  const requestId = useRef(0);
  const rpc = (method: string, params: unknown) => new Promise<any>((resolve, reject) => {
    const id = ++requestId.current;
    const listener = (event: MessageEvent) => {
      const message = JSON.parse(event.data);
      if (message.id !== id) return;
      socket.current?.removeEventListener("message", listener);
      message.error ? reject(new Error(message.error.message)) : resolve(message.result);
    };
    socket.current?.addEventListener("message", listener);
    socket.current?.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
  });
  useEffect(() => {
    const connection = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/rpc");
    socket.current = connection;
    connection.onopen = () => {
      setStatus("已连接"); void rpc("system.ping", {});
      void rpc("threads.list", {}).then((result) => setThreads(result.data ?? result.threads ?? [])).catch((reason) => setError(String(reason)));
    };
    connection.onclose = () => setStatus("离线");
    connection.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.method !== "codex.event") return;
      if (message.params?.method === "turn/completed") { setBusy(false); return; }
      if (message.params?.method !== "item/agentMessage/delta") return;
      const delta = message.params.params?.delta ?? "";
      setMessages((items) => items.at(-1)?.role === "assistant"
        ? [...items.slice(0, -1), { role: "assistant", text: items.at(-1)!.text + delta }]
        : [...items, { role: "assistant", text: delta }]);
    };
    return () => connection.close();
  }, []);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!input.trim() || socket.current?.readyState !== WebSocket.OPEN) return;
    const text = input; setInput(""); setBusy(true); setError(undefined); setMessages((items) => [...items, { role: "user", text }]);
    try {
      let current = threadId;
      if (!current) { const result = await rpc("threads.create", { cwd: "F:\\Codex Project\\Justin-Agent\\Justin-Agent" }); current = result.thread.id; setThreadId(current); setThreads((items) => [{ id: current!, preview: "新任务" }, ...items]); }
      await rpc("threads.send", { threadId: current, input: [{ type: "text", text }] });
    } catch (error) { const message = error instanceof Error ? error.message : "请求失败"; setError(message); setBusy(false); setMessages((items) => [...items, { role: "assistant", text: message }]); }
  };
  const selectThread = async (id: string) => { setThreadId(id); setMessages([]); try { await rpc("threads.resume", { threadId: id }); } catch (reason) { setError(String(reason)); } };
  const interrupt = async () => { if (!threadId) return; try { await rpc("turns.interrupt", { threadId }); } catch (reason) { setError(String(reason)); } };
  return <main>
    <aside><header>Justin Agent <span>{status}</span></header><h2>工作区</h2><button className="workspace">+ F:\\Codex Project</button><h2>线程</h2><button className={!threadId ? "thread active" : "thread"} onClick={() => { setThreadId(undefined); setMessages([]); }}>+ 新建任务</button>{threads.map((thread) => <button className={thread.id === threadId ? "thread active" : "thread"} onClick={() => void selectThread(thread.id)} key={thread.id}>{thread.name ?? thread.preview ?? thread.id}</button>)}</aside>
    <section className="conversation"><div className={messages.length ? "messages" : "empty"}>{messages.length ? messages.map((item, index) => <p className={item.role} key={index}>{item.text}</p>) : <><span>✦</span><h1>Codex 内核已就绪</h1><p>{error ?? "选择工作区后开始一个任务。代理执行、审批和会话由 Codex App Server 管理。"}</p></>}</div>
      <form onSubmit={submit}><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="告诉 Justin 你想完成什么…"/>{busy ? <button type="button" onClick={() => void interrupt()}>■</button> : <button disabled={status !== "已连接"}>↑</button>}</form></section>
    <aside className="inspector"><h2>计划</h2><p className="muted">—</p><nav>{["终端", "工具", "变更", "审批"].map((item) => <button key={item}>{item}</button>)}</nav></aside>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
