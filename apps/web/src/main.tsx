import { FormEvent, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type ChatMessage = { role: "user" | "assistant"; text: string };
function App() {
  const [threadId, setThreadId] = useState<string>();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState("离线");
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
    connection.onopen = () => { setStatus("已连接"); void rpc("system.ping", {}); };
    connection.onclose = () => setStatus("离线");
    connection.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.method !== "codex.event" || message.params?.method !== "item/agentMessage/delta") return;
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
    const text = input; setInput(""); setMessages((items) => [...items, { role: "user", text }]);
    try {
      let current = threadId;
      if (!current) { const result = await rpc("threads.create", { cwd: "F:\\Codex Project\\Justin-Agent\\Justin-Agent" }); current = result.thread.id; setThreadId(current); }
      await rpc("threads.send", { threadId: current, input: [{ type: "text", text }] });
    } catch (error) { setMessages((items) => [...items, { role: "assistant", text: error instanceof Error ? error.message : "请求失败" }]); }
  };
  return <main>
    <aside><header>Justin Agent <span>{status}</span></header><h2>工作区</h2><button className="workspace">+ F:\\Codex Project</button><h2>线程</h2><button className="thread active">{threadId ?? "新建任务"}</button></aside>
    <section className="conversation"><div className={messages.length ? "messages" : "empty"}>{messages.length ? messages.map((item, index) => <p className={item.role} key={index}>{item.text}</p>) : <><span>✦</span><h1>Codex 内核已就绪</h1><p>选择工作区后开始一个任务。代理执行、审批和会话由 Codex App Server 管理。</p></>}</div>
      <form onSubmit={submit}><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="告诉 Justin 你想完成什么…"/><button disabled={status !== "已连接"}>↑</button></form></section>
    <aside className="inspector"><h2>计划</h2><p className="muted">—</p><nav>{["终端", "工具", "变更", "审批"].map((item) => <button key={item}>{item}</button>)}</nav></aside>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
