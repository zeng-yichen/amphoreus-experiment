"use client";

import { useParams } from "next/navigation";
import { useState, useRef, useEffect, useCallback } from "react";
import { strategyApi, reportApi } from "@/lib/api";
import Link from "next/link";

interface TerminalLine {
  type: string;
  text: string;
  timestamp: number;
}

export default function StrategyPage() {
  const params = useParams();
  const company = params.company as string;
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [lines, setLines] = useState<TerminalLine[]>([]);
  const [currentStrategy, setCurrentStrategy] = useState<string | null>(null);
  const [loadingStrategy, setLoadingStrategy] = useState(true);
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [reportWeeks, setReportWeeks] = useState(2);
  const [activeView, setActiveView] = useState<"strategy" | "report" | "terminal">("strategy");
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  const loadCurrentStrategy = useCallback(async () => {
    setLoadingStrategy(true);
    try {
      // Prefer HTML version for proper rendering; fall back to markdown
      const htmlRes = await strategyApi.getHtml(company);
      if (htmlRes.html) {
        setCurrentStrategy(htmlRes.html);
      } else {
        const mdRes = await strategyApi.getCurrent(company);
        setCurrentStrategy(mdRes.strategy);
      }
    } catch {
      setCurrentStrategy(null);
    } finally {
      setLoadingStrategy(false);
    }
  }, [company]);

  useEffect(() => {
    loadCurrentStrategy();
  }, [loadCurrentStrategy]);

  const loadReport = useCallback(async (weeks: number) => {
    setLoadingReport(true);
    try {
      const res = await reportApi.getHtml(company, weeks);
      setReportHtml(res.html);
    } catch {
      setReportHtml(null);
    } finally {
      setLoadingReport(false);
    }
  }, [company]);

  useEffect(() => {
    if (activeView === "report" && !reportHtml && !loadingReport) {
      loadReport(reportWeeks);
    }
  }, [activeView, reportHtml, loadingReport, loadReport, reportWeeks]);

  async function handleGenerate() {
    if (isGenerating) return;
    setIsGenerating(true);
    setLines([]);
    setActiveView("terminal");

    try {
      const { job_id } = await strategyApi.generate(company, prompt || undefined);
      setLines((prev) => [
        ...prev,
        { type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() },
      ]);

      for await (const data of strategyApi.streamJob(job_id)) {
        const text = extractText(data);
        setLines((prev) => [...prev, {
          type: data.type,
          text,
          timestamp: (data.timestamp || Date.now() / 1000) * 1000,
        }]);

        if (data.type === "done") {
          setIsGenerating(false);
          await loadCurrentStrategy();
          setActiveView("strategy");
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
        <h1 className="text-lg font-semibold">Content Strategy</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>
        <div className="flex-1" />
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Optional focus prompt..."
          className="w-80 rounded-lg border border-stone-200 px-3 py-1.5 text-sm focus:border-stone-400 focus:outline-none"
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
        />
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-50"
        >
          {isGenerating ? "Generating..." : "Generate Strategy"}
        </button>
      </header>

      <div className="flex border-b border-stone-200 bg-stone-50 px-6">
        {(["strategy", "report", "terminal"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveView(tab)}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeView === tab
                ? "border-stone-900 text-stone-900"
                : "border-transparent text-stone-500 hover:text-stone-700"
            }`}
          >
            {tab === "strategy" ? "Current Strategy" : tab === "report" ? "Progress Report" : "Generation Progress"}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto" ref={terminalRef}>
        {activeView === "strategy" && (
          <div className="mx-auto max-w-4xl p-8">
            {loadingStrategy ? (
              <p className="text-stone-500">Loading strategy...</p>
            ) : currentStrategy ? (
              currentStrategy.trimStart().startsWith("<!DOCTYPE") || currentStrategy.trimStart().startsWith("<html") ? (
                <div
                  className="strategy-embed"
                  dangerouslySetInnerHTML={{ __html: currentStrategy }}
                />
              ) : (
                <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-stone-800">
                  {currentStrategy}
                </pre>
              )
            ) : (
              <div className="rounded-lg border border-dashed border-stone-300 p-8 text-center">
                <p className="text-stone-500">No content strategy found for {company}.</p>
                <p className="mt-1 text-sm text-stone-400">
                  Click Generate Strategy to create one from interview transcripts and post data.
                </p>
              </div>
            )}
          </div>
        )}

        {activeView === "report" && (
          <div className="mx-auto max-w-5xl p-8">
            <div className="mb-4 flex items-center gap-3">
              <label className="text-sm text-stone-500">Window:</label>
              <select
                value={reportWeeks}
                onChange={(e) => {
                  const w = Number(e.target.value);
                  setReportWeeks(w);
                  setReportHtml(null);
                  loadReport(w);
                }}
                className="rounded border border-stone-200 px-2 py-1 text-sm"
              >
                <option value={1}>1 week</option>
                <option value={2}>2 weeks</option>
                <option value={3}>3 weeks</option>
                <option value={4}>4 weeks</option>
              </select>
              <button
                onClick={() => { setReportHtml(null); loadReport(reportWeeks); }}
                disabled={loadingReport}
                className="rounded-lg bg-stone-900 px-3 py-1 text-xs font-medium text-white hover:bg-stone-800 disabled:opacity-50"
              >
                {loadingReport ? "Generating..." : "Refresh"}
              </button>
            </div>
            {loadingReport ? (
              <div className="flex items-center gap-2 py-12 text-stone-500">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                  <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="4" strokeLinecap="round" className="opacity-75" />
                </svg>
                Generating progress report...
              </div>
            ) : reportHtml ? (
              <div
                className="report-embed rounded-lg border border-stone-200 bg-white"
                dangerouslySetInnerHTML={{ __html: reportHtml }}
              />
            ) : (
              <div className="rounded-lg border border-dashed border-stone-300 p-8 text-center">
                <p className="text-stone-500">No report data available for {company}.</p>
                <p className="mt-1 text-sm text-stone-400">
                  Click Refresh to generate a progress report from Ordinal analytics and learning data.
                </p>
              </div>
            )}
          </div>
        )}

        {activeView === "terminal" && (
          <div className="space-y-0.5 bg-stone-950 p-4 font-mono text-sm">
            {lines.length === 0 && (
              <p className="text-stone-500">Click Generate Strategy to start.</p>
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
      </div>
    </div>
  );
}

function extractText(data: Record<string, unknown>): string {
  const d = data.data as Record<string, unknown> | undefined;
  if (!d) return JSON.stringify(data);
  if (data.type === "done") return (d.output as string)?.slice(0, 200) || "Strategy complete.";
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
