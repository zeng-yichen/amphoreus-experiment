"use client";

import { useParams } from "next/navigation";
import { useState, useRef, useEffect, useCallback } from "react";
import { ghostwriterApi, postsApi, imagesApi } from "@/lib/api";
import Link from "next/link";

interface TerminalLine {
  type: string;
  text: string;
  timestamp: number;
}

interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
}

interface RunEntry {
  id: string;
  agent: string;
  status: string;
  prompt: string | null;
  output: string | null;
  error: string | null;
  created_at: number;
  completed_at: number | null;
}

interface RunEvent {
  id: number;
  run_id: string;
  event_type: string;
  data: string | null;
  timestamp: number;
}

export default function GhostwriterIDE() {
  const params = useParams();
  const company = params.company as string;
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [activeTab, setActiveTab] = useState<"terminal" | "files" | "history" | "posts">("terminal");
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [filePath, setFilePath] = useState("");
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [selectedRun, setSelectedRun] = useState<RunEntry | null>(null);
  const [runEvents, setRunEvents] = useState<RunEvent[]>([]);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [posts, setPosts] = useState<any[]>([]);
  const [loadingPosts, setLoadingPosts] = useState(false);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines, runEvents]);

  const loadFiles = useCallback(async (path = "") => {
    setLoadingFiles(true);
    try {
      const res = await ghostwriterApi.getFiles(company, path);
      setFiles(res.files);
      setFilePath(path);
    } catch {
      setFiles([]);
    } finally {
      setLoadingFiles(false);
    }
  }, [company]);

  const loadRuns = useCallback(async () => {
    setLoadingRuns(true);
    try {
      const res = await ghostwriterApi.getRuns(company);
      setRuns(res.runs);
    } catch {
      setRuns([]);
    } finally {
      setLoadingRuns(false);
    }
  }, [company]);

  const loadPosts = useCallback(async () => {
    setLoadingPosts(true);
    try {
      const res = await postsApi.list(company);
      setPosts(res.posts);
    } catch {
      setPosts([]);
    } finally {
      setLoadingPosts(false);
    }
  }, [company]);

  const loadRunDetail = useCallback(async (run: RunEntry) => {
    setSelectedRun(run);
    setLoadingEvents(true);
    try {
      const res = await ghostwriterApi.getRunEvents(run.id);
      setRunEvents(res.events);
    } catch {
      setRunEvents([]);
    } finally {
      setLoadingEvents(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "files") loadFiles(filePath);
    if (activeTab === "history") { loadRuns(); setSelectedRun(null); }
    if (activeTab === "posts") loadPosts();
  }, [activeTab, loadFiles, loadRuns, loadPosts, filePath]);

  async function handleGenerate() {
    if (isGenerating) return;
    setIsGenerating(true);
    setLines([]);
    setActiveTab("terminal");

    try {
      const { job_id } = await ghostwriterApi.generate(company, prompt || undefined);
      setLines((prev) => [
        ...prev,
        { type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() },
      ]);

      for await (const data of ghostwriterApi.streamJob(job_id)) {
        const text = extractText(data);
        setLines((prev) => [...prev, { type: data.type, text, timestamp: (data.timestamp || Date.now() / 1000) * 1000 }]);

        if (data.type === "done" || data.type === "error") {
          setIsGenerating(false);
        }
      }
      setIsGenerating(false);
    } catch (e) {
      setLines((prev) => [
        ...prev,
        { type: "error", text: String(e), timestamp: Date.now() },
      ]);
      setIsGenerating(false);
    }
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-4 border-b border-stone-200 bg-white px-6 py-3">
        <Link href="/home" className="text-sm text-stone-400 hover:text-stone-600">&larr;</Link>
        <h1 className="text-lg font-semibold">Ghostwriter</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>
        <div className="flex-1" />
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Optional prompt..."
          className="w-96 rounded-lg border border-stone-200 px-3 py-1.5 text-sm focus:border-stone-400 focus:outline-none"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-50"
        >
          {isGenerating ? "Generating..." : "Generate"}
        </button>
      </header>

      <div className="flex border-b border-stone-200 bg-stone-50 px-6">
        {(["terminal", "files", "posts", "history"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab
                ? "border-stone-900 text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-700"
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto bg-stone-950 p-4 font-mono text-sm" ref={terminalRef}>
        {activeTab === "terminal" && (
          <div className="space-y-0.5">
            {lines.length === 0 && (
              <p className="text-stone-500">Ready. Click Generate to start.</p>
            )}
            {lines.map((line, i) => (
              <div key={i} className={getLineColor(line.type)}>
                <span className="mr-2 text-stone-600">
                  {new Date(line.timestamp).toLocaleTimeString()}
                </span>
                <span className="mr-2 text-stone-500">[{line.type}]</span>
                <span className="whitespace-pre-wrap">{line.text}</span>
              </div>
            ))}
          </div>
        )}

        {activeTab === "files" && (
          <div className="space-y-1">
            <div className="mb-3 flex items-center gap-2 text-stone-400">
              <span className="text-xs">workspace/{filePath || "."}</span>
              {filePath && (
                <button
                  onClick={() => loadFiles(filePath.split("/").slice(0, -1).join("/"))}
                  className="rounded px-2 py-0.5 text-xs text-stone-400 hover:bg-stone-800 hover:text-stone-200"
                >
                  ..
                </button>
              )}
              <button
                onClick={() => loadFiles(filePath)}
                className="ml-auto rounded px-2 py-0.5 text-xs text-stone-500 hover:bg-stone-800 hover:text-stone-200"
              >
                Refresh
              </button>
            </div>
            {loadingFiles ? (
              <p className="text-stone-500">Loading...</p>
            ) : files.length === 0 ? (
              <p className="text-stone-500">No files. Provision workspace first or run a generation.</p>
            ) : (
              files.map((f) => (
                <button
                  key={f.path}
                  onClick={() => f.is_dir && loadFiles(f.path)}
                  className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm ${
                    f.is_dir
                      ? "text-blue-400 hover:bg-stone-800"
                      : "cursor-default text-stone-300"
                  }`}
                >
                  <span className="w-4 text-center text-xs text-stone-500">{f.is_dir ? "D" : "F"}</span>
                  <span className="flex-1">{f.name}</span>
                  {f.size !== null && (
                    <span className="text-xs text-stone-600">{formatBytes(f.size)}</span>
                  )}
                </button>
              ))
            )}
          </div>
        )}

        {activeTab === "history" && !selectedRun && (
          <div className="space-y-2">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs text-stone-400">Run history for {company}</span>
              <button
                onClick={loadRuns}
                className="rounded px-2 py-0.5 text-xs text-stone-500 hover:bg-stone-800 hover:text-stone-200"
              >
                Refresh
              </button>
            </div>
            {loadingRuns ? (
              <p className="text-stone-500">Loading...</p>
            ) : runs.length === 0 ? (
              <p className="text-stone-500">No runs yet.</p>
            ) : (
              runs.map((r) => (
                <button
                  key={r.id}
                  onClick={() => loadRunDetail(r)}
                  className="w-full rounded border border-stone-800 bg-stone-900 px-3 py-2 text-left text-sm transition-colors hover:border-stone-600 hover:bg-stone-800"
                >
                  <div className="flex items-center gap-3">
                    <span className={`text-xs font-medium ${statusColor(r.status)}`}>
                      {r.status}
                    </span>
                    <span className="text-xs text-stone-500">{r.agent}</span>
                    <span className="ml-auto text-xs text-stone-600">
                      {new Date(r.created_at * 1000).toLocaleString()}
                    </span>
                    {r.completed_at && (
                      <span className="text-xs text-stone-600">
                        ({Math.round(r.completed_at - r.created_at)}s)
                      </span>
                    )}
                  </div>
                  {r.prompt && (
                    <p className="mt-1 truncate text-xs text-stone-400">{r.prompt}</p>
                  )}
                  {r.error && (
                    <p className="mt-1 truncate text-xs text-red-400">{r.error}</p>
                  )}
                </button>
              ))
            )}
          </div>
        )}

        {activeTab === "history" && selectedRun && (
          <RunDetailView
            run={selectedRun}
            events={runEvents}
            loading={loadingEvents}
            onBack={() => setSelectedRun(null)}
          />
        )}

        {activeTab === "posts" && (
          <PostsManager
            company={company}
            posts={posts}
            loading={loadingPosts}
            actionInProgress={actionInProgress}
            onAction={setActionInProgress}
            onRefresh={loadPosts}
          />
        )}
      </div>
    </div>
  );
}

function RunDetailView({
  run,
  events,
  loading,
  onBack,
}: {
  run: RunEntry;
  events: RunEvent[];
  loading: boolean;
  onBack: () => void;
}) {
  const toolCalls = events.filter((e) => e.event_type === "tool_call");
  const errors = events.filter((e) => e.event_type === "error");
  const duration = run.completed_at ? Math.round(run.completed_at - run.created_at) : null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="rounded px-2 py-1 text-xs text-stone-400 hover:bg-stone-800 hover:text-stone-200"
        >
          &larr; Back
        </button>
        <span className={`text-xs font-medium ${statusColor(run.status)}`}>{run.status}</span>
        <span className="text-xs text-stone-500">{run.agent}</span>
        <span className="text-xs text-stone-600">{new Date(run.created_at * 1000).toLocaleString()}</span>
        {duration !== null && <span className="text-xs text-stone-600">({duration}s)</span>}
      </div>

      <div className="flex gap-4 text-xs text-stone-500">
        <span>{events.length} events</span>
        <span>{toolCalls.length} tool calls</span>
        {errors.length > 0 && <span className="text-red-400">{errors.length} errors</span>}
      </div>

      {run.prompt && (
        <div className="rounded border border-stone-800 bg-stone-900 p-3">
          <span className="text-xs text-stone-500">Prompt: </span>
          <span className="text-xs text-stone-300">{run.prompt}</span>
        </div>
      )}

      {loading ? (
        <p className="text-stone-500">Loading events...</p>
      ) : events.length === 0 ? (
        <p className="text-stone-500">No events recorded for this run.</p>
      ) : (
        <div className="space-y-0.5">
          {events.map((evt) => {
            const parsed = parseEventData(evt);
            return (
              <div key={evt.id} className={`flex gap-2 ${getLineColor(evt.event_type)}`}>
                <span className="shrink-0 text-stone-600">
                  {new Date(evt.timestamp * 1000).toLocaleTimeString()}
                </span>
                <span className="shrink-0 text-stone-500">[{evt.event_type}]</span>
                <span className="whitespace-pre-wrap break-all">{parsed}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function parseEventData(evt: RunEvent): string {
  if (!evt.data) return "";
  try {
    const d = JSON.parse(evt.data);
    switch (evt.event_type) {
      case "tool_call":
        return `${d.name || ""}(${d.arguments?.summary || d.arguments || ""})`;
      case "tool_result":
        return `${d.name || ""} -> ${(d.result || "").slice(0, 300)}`;
      case "thinking":
        return d.text || "";
      case "text_delta":
        return d.text || "";
      case "error":
        return d.message || JSON.stringify(d);
      case "done":
        return (d.output || "").slice(0, 500) || "Generation complete.";
      case "status":
        return d.message || "";
      case "compaction":
        return d.message || "Context compaction";
      default:
        return JSON.stringify(d).slice(0, 300);
    }
  } catch {
    return evt.data.slice(0, 300);
  }
}

function extractText(data: Record<string, unknown>): string {
  const d = data.data as Record<string, unknown> | undefined;
  if (!d) return JSON.stringify(data);

  if (data.type === "done") return (d.output as string) || "Generation complete.";
  if (data.type === "error") return (d.message as string) || "Unknown error";
  if (data.type === "tool_call") return `${d.name}(${(d.arguments as Record<string, unknown>)?.summary || d.arguments || ""})`;
  if (data.type === "tool_result") return `${d.name} -> ${((d.result as string) || "").slice(0, 200)}`;

  return (d.text as string) || (d.message as string) || (d.name as string) || JSON.stringify(d);
}

function statusColor(status: string): string {
  if (status === "completed") return "text-green-400";
  if (status === "failed") return "text-red-400";
  return "text-yellow-400";
}

function getLineColor(type: string): string {
  switch (type) {
    case "thinking": return "text-blue-400";
    case "tool_call": return "text-amber-400";
    case "tool_result": return "text-emerald-400";
    case "text_delta": return "text-stone-200";
    case "compaction": return "text-purple-400";
    case "error": return "text-red-400";
    case "done": return "text-green-400";
    case "status": return "text-cyan-400";
    default: return "text-stone-400";
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

function PostsManager({
  company,
  posts,
  loading,
  actionInProgress,
  onAction,
  onRefresh,
}: {
  company: string;
  posts: any[];
  loading: boolean;
  actionInProgress: string | null;
  onAction: (id: string | null) => void;
  onRefresh: () => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [factReport, setFactReport] = useState<{ id: string; report: string } | null>(null);
  const [imageJobId, setImageJobId] = useState<string | null>(null);
  const [imageLines, setImageLines] = useState<{ type: string; text: string }[]>([]);
  const [generatingImageFor, setGeneratingImageFor] = useState<string | null>(null);

  async function handleGenerateImage(post: any) {
    setGeneratingImageFor(post.id);
    setImageLines([]);
    setImageJobId(null);
    try {
      const { job_id } = await imagesApi.generate(company, post.content);
      setImageJobId(job_id);
      for await (const data of imagesApi.streamJob(job_id)) {
        const d = data.data || {};
        const text =
          d.text || d.message || d.name || d.output?.slice?.(0, 200) || JSON.stringify(d);
        setImageLines((prev) => [...prev, { type: data.type, text }]);
        if (data.type === "done" || data.type === "error") {
          setGeneratingImageFor(null);
        }
      }
      setGeneratingImageFor(null);
    } catch (e) {
      setImageLines((prev) => [...prev, { type: "error", text: String(e) }]);
      setGeneratingImageFor(null);
    }
  }

  async function handleDelete(postId: string) {
    if (!confirm("Delete this post?")) return;
    onAction(postId);
    try {
      await postsApi.delete(postId);
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  async function handleRewrite(post: any) {
    onAction(post.id);
    try {
      const res = await postsApi.rewrite(post.id, company, post.content);
      onRefresh();
      if (res.result) {
        setEditingId(post.id);
        setEditText(typeof res.result === "string" ? res.result : res.result.content || "");
      }
    } finally {
      onAction(null);
    }
  }

  async function handleFactCheck(post: any) {
    onAction(post.id);
    try {
      const res = await postsApi.factCheck(post.id, company, post.content);
      setFactReport({ id: post.id, report: res.report });
    } finally {
      onAction(null);
    }
  }

  async function handleSaveEdit(postId: string) {
    onAction(postId);
    try {
      await postsApi.update(postId, company, editText);
      setEditingId(null);
      setEditText("");
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  async function handlePush(post: any) {
    onAction(post.id);
    try {
      await postsApi.push(company, post.content);
    } finally {
      onAction(null);
    }
  }

  if (loading) return <p className="text-stone-500">Loading posts...</p>;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-stone-300">
          {posts.length} post{posts.length !== 1 ? "s" : ""}
        </h3>
        <button onClick={onRefresh} className="text-xs text-stone-500 hover:text-stone-300">
          Refresh
        </button>
      </div>

      {posts.length === 0 && (
        <p className="text-sm text-stone-500">
          No posts yet. Generate some with the Ghostwriter.
        </p>
      )}

      {factReport && (
        <div className="rounded-lg border border-cyan-800 bg-cyan-950/30 p-4">
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-sm font-medium text-cyan-300">Fact-Check Report</h4>
            <button onClick={() => setFactReport(null)} className="text-xs text-stone-500 hover:text-stone-300">
              Dismiss
            </button>
          </div>
          <pre className="whitespace-pre-wrap text-xs text-stone-300">{factReport.report}</pre>
        </div>
      )}

      {imageLines.length > 0 && (
        <div className="rounded-lg border border-amber-800 bg-amber-950/20 p-4">
          <div className="mb-2 flex items-center justify-between">
            <h4 className="text-sm font-medium text-amber-300">
              Image Assembly {generatingImageFor ? "(running...)" : "(complete)"}
            </h4>
            <button onClick={() => setImageLines([])} className="text-xs text-stone-500 hover:text-stone-300">
              Dismiss
            </button>
          </div>
          <div className="max-h-48 space-y-0.5 overflow-y-auto font-mono text-xs">
            {imageLines.map((line, i) => (
              <div key={i} className={
                line.type === "error" ? "text-red-400" :
                line.type === "tool_call" ? "text-amber-400" :
                line.type === "tool_result" ? "text-emerald-400" :
                line.type === "done" ? "text-green-400" :
                line.type === "status" ? "text-cyan-400" :
                "text-stone-400"
              }>
                <span className="mr-1 text-stone-600">[{line.type}]</span>
                {line.text}
              </div>
            ))}
          </div>
        </div>
      )}

      {posts.map((post) => (
        <div
          key={post.id}
          className="rounded-lg border border-stone-800 bg-stone-900 p-4"
        >
          {editingId === post.id ? (
            <div className="space-y-2">
              <textarea
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                rows={8}
                className="w-full rounded border border-stone-700 bg-stone-950 p-3 text-sm text-stone-200 focus:border-stone-500 focus:outline-none"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => handleSaveEdit(post.id)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-700 px-3 py-1 text-xs text-white hover:bg-stone-600 disabled:opacity-50"
                >
                  Save
                </button>
                <button
                  onClick={() => { setEditingId(null); setEditText(""); }}
                  className="rounded bg-stone-800 px-3 py-1 text-xs text-stone-400 hover:bg-stone-700"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <>
              <pre className="mb-3 whitespace-pre-wrap text-sm text-stone-200">
                {post.content?.slice(0, 600)}
                {post.content?.length > 600 ? "..." : ""}
              </pre>
              <div className="flex flex-wrap items-center gap-2 border-t border-stone-800 pt-3">
                <button
                  onClick={() => { setEditingId(post.id); setEditText(post.content || ""); }}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleRewrite(post)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700 disabled:opacity-50"
                >
                  {actionInProgress === post.id ? "..." : "Rewrite"}
                </button>
                <button
                  onClick={() => handleFactCheck(post)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700 disabled:opacity-50"
                >
                  {actionInProgress === post.id ? "..." : "Fact-check"}
                </button>
                <button
                  onClick={() => handleGenerateImage(post)}
                  disabled={generatingImageFor === post.id || actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-amber-400 hover:bg-stone-700 disabled:opacity-50"
                >
                  {generatingImageFor === post.id ? "Generating..." : "Generate Image"}
                </button>
                <button
                  onClick={() => handlePush(post)}
                  disabled={actionInProgress === post.id}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-cyan-400 hover:bg-stone-700 disabled:opacity-50"
                >
                  Push to Calendar
                </button>
                <div className="flex-1" />
                <span className="text-xs text-stone-600">
                  {post.status || "draft"}
                  {post.created_at ? ` \u00b7 ${new Date(post.created_at * 1000).toLocaleDateString()}` : ""}
                </span>
                <button
                  onClick={() => handleDelete(post.id)}
                  disabled={actionInProgress === post.id}
                  className="rounded px-2 py-1 text-xs text-red-500 hover:bg-red-950 disabled:opacity-50"
                >
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}
