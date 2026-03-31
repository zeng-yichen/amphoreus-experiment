"use client";

import { useParams } from "next/navigation";
import { useState, useRef, useEffect } from "react";
import { briefingsApi } from "@/lib/api";
import Link from "next/link";

interface TerminalLine {
  type: string;
  text: string;
  timestamp: number;
}

export default function BriefingsPage() {
  const params = useParams();
  const company = params.company as string;
  const [clientName, setClientName] = useState(company);
  const [isGenerating, setIsGenerating] = useState(false);
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [briefingOutput, setBriefingOutput] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<"terminal" | "briefing">("terminal");
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  async function handleGenerate() {
    if (isGenerating) return;
    setIsGenerating(true);
    setLines([]);
    setBriefingOutput(null);
    setActiveView("terminal");

    try {
      const { job_id } = await briefingsApi.generate(clientName, company);
      setLines((prev) => [
        ...prev,
        { type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() },
      ]);

      for await (const data of briefingsApi.streamJob(job_id)) {
        const text = extractText(data);
        setLines((prev) => [...prev, {
          type: data.type,
          text,
          timestamp: (data.timestamp || Date.now() / 1000) * 1000,
        }]);

        if (data.type === "done") {
          const output = data.data?.output || "";
          if (output) setBriefingOutput(output);
          setIsGenerating(false);
          setActiveView("briefing");
        }
        if (data.type === "error") {
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
        <h1 className="text-lg font-semibold">Interview Briefing</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>
        <div className="flex-1" />
        <input
          value={clientName}
          onChange={(e) => setClientName(e.target.value)}
          placeholder="Client name..."
          className="w-64 rounded-lg border border-stone-200 px-3 py-1.5 text-sm focus:border-stone-400 focus:outline-none"
        />
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-50"
        >
          {isGenerating ? "Generating..." : "Generate Briefing"}
        </button>
      </header>

      <div className="flex border-b border-stone-200 bg-stone-50 px-6">
        {(["terminal", "briefing"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveView(tab)}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeView === tab
                ? "border-stone-900 text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-700"
            }`}
          >
            {tab === "terminal" ? "Progress" : "Briefing"}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto bg-stone-950 p-4 font-mono text-sm" ref={terminalRef}>
        {activeView === "terminal" && (
          <div className="space-y-0.5">
            {lines.length === 0 && (
              <p className="text-stone-500">Ready. Click Generate Briefing to start.</p>
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

        {activeView === "briefing" && (
          <div className="prose prose-invert max-w-none">
            {briefingOutput ? (
              <pre className="whitespace-pre-wrap text-stone-200">{briefingOutput}</pre>
            ) : (
              <p className="text-stone-500">No briefing generated yet. Run a generation first.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function extractText(data: Record<string, unknown>): string {
  const d = data.data as Record<string, unknown> | undefined;
  if (!d) return JSON.stringify(data);
  if (data.type === "done") return (d.output as string)?.slice(0, 200) || "Briefing complete.";
  if (data.type === "error") return (d.message as string) || "Unknown error";
  if (data.type === "tool_call") return `${d.name}(${(d.arguments as Record<string, unknown>)?.summary || d.arguments || ""})`;
  if (data.type === "tool_result") return `${d.name} -> ${((d.result as string) || "").slice(0, 200)}`;
  return (d.text as string) || (d.message as string) || (d.name as string) || JSON.stringify(d);
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
