"use client";

import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { interviewApi, briefingsApi, strategyApi, API_BASE } from "@/lib/api";


interface TranscriptSegment {
  text: string;
  timestamp: number;
}

interface Suggestion {
  text: string;
  timestamp: number;
}

export default function InterviewPage() {
  const params = useParams();
  const router = useRouter();
  const company = params.company as string;

  const [clientName, setClientName] = useState(company);

  function handleCompanyChange(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      const slug = (e.target as HTMLInputElement).value.trim().toLowerCase().replace(/\s+/g, "-");
      if (slug && slug !== company) router.push(`/interview/${slug}`);
    }
  }
  const [isRecording, setIsRecording] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("Ready");
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [trashed, setTrashed] = useState(false);
  const [hasBlackhole, setHasBlackhole] = useState<boolean | null>(null);
  const [hasBriefing, setHasBriefing] = useState<boolean | null>(null);
  const [briefingContent, setBriefingContent] = useState<string | null>(null);
  const [hasStrategyHtml, setHasStrategyHtml] = useState(false);
  const [strategyMarkdown, setStrategyMarkdown] = useState<string | null>(null);
  const [strategyGenerating, setStrategyGenerating] = useState(false);
  const [strategyLog, setStrategyLog] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const transcriptRef = useRef<HTMLDivElement>(null);

  // Detect BlackHole, briefing existence, and briefing content on mount
  useEffect(() => {
    interviewApi
      .listDevices()
      .then((res) => setHasBlackhole(res.has_blackhole))
      .catch(() => setHasBlackhole(false));

    briefingsApi
      .check(company)
      .then((res) => {
        setHasBriefing(res.exists);
        if (res.exists) {
          briefingsApi
            .get(company)
            .then((r) => setBriefingContent(r.content))
            .catch(() => setBriefingContent(null));
        }
      })
      .catch(() => setHasBriefing(false));

    strategyApi
      .getCurrent(company)
      .then((res) => setStrategyMarkdown(res.strategy))
      .catch(() => setStrategyMarkdown(null));

    strategyApi
      .getHtml(company)
      .then((res) => setHasStrategyHtml(!!res.html))
      .catch(() => setHasStrategyHtml(false));
  }, [company]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  async function handleGenerateStrategy() {
    setStrategyGenerating(true);
    setStrategyLog("");
    try {
      const { job_id } = await strategyApi.generate(company);
      for await (const data of strategyApi.streamJob(job_id)) {
        if (data.type === "text_delta") {
          setStrategyLog((prev) => prev + ((data.data as any)?.text || ""));
        } else if (data.type === "done") {
          const [mdRes, htmlRes] = await Promise.all([
            strategyApi.getCurrent(company),
            strategyApi.getHtml(company),
          ]);
          setStrategyMarkdown(mdRes.strategy);
          setHasStrategyHtml(!!htmlRes.html);
          break;
        } else if (data.type === "error") {
          setStrategyLog((prev) => prev + `\n\nError: ${(data.data as any)?.message}`);
          break;
        }
      }
    } catch (e) {
      setStrategyLog((prev) => prev + `\n\nFailed: ${String(e)}`);
    } finally {
      setStrategyGenerating(false);
    }
  }

  async function handleStart() {
    if (isRecording) return;
    setIsRecording(true);
    setTranscript([]);
    setSuggestions([]);
    setSavedPath(null);
    setTrashed(false);
    setError(null);
    setStatus("Starting...");

    try {
      const { job_id } = await interviewApi.start(company, clientName);
      setJobId(job_id);
      setStatus("Connecting...");

      for await (const data of interviewApi.streamJob(job_id)) {
        const ts = (data.timestamp || Date.now() / 1000) * 1000;

        if (data.type === "status") {
          setStatus((data.data as any)?.message || "");
        } else if (data.type === "text_delta") {
          const text = (data.data as any)?.text || "";
          if (text) setTranscript((prev) => [...prev, { text, timestamp: ts }]);
        } else if (data.type === "tool_result") {
          const suggestion = (data.data as any)?.result || "";
          if (suggestion) {
            setSuggestions((prev) => [{ text: suggestion, timestamp: ts }, ...prev]);
          }
        } else if (data.type === "error") {
          setError((data.data as any)?.message || "Unknown error");
          setIsRecording(false);
          setStatus("Error");
          return;
        } else if (data.type === "done") {
          const filePath = (data.data as any)?.output || null;
          const msg = (data.data as any)?.message || "Session complete";
          setSavedPath(filePath);
          setIsRecording(false);
          setStatus(msg);
          return;
        }
      }
      setIsRecording(false);
    } catch (e) {
      setError(String(e));
      setIsRecording(false);
      setStatus("Error");
    }
  }

  async function handleStop() {
    if (!isRecording || !jobId) return;
    setStatus("Stopping — saving transcript...");
    try {
      await interviewApi.stop(jobId, company);
    } catch {
      // Backend signals the session; done event will arrive via SSE
    }
  }

  const canStart = hasBlackhole !== false && hasBriefing === true;

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-stone-200 bg-white px-6 py-3">
        <Link href="/home" className="text-sm text-stone-400 hover:text-stone-600">
          &larr;
        </Link>
        <h1 className="text-lg font-semibold">Interview Companion</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>

        {isRecording && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-red-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
            Recording
          </span>
        )}

        <div className="flex-1" />

        <input
          value={clientName}
          onChange={(e) => setClientName(e.target.value)}
          onKeyDown={handleCompanyChange}
          disabled={isRecording}
          placeholder="Company keyword — press Enter"
          title="Type a company keyword and press Enter to switch clients"
          className="w-64 rounded-lg border border-stone-200 px-3 py-1.5 text-sm focus:border-stone-400 focus:outline-none disabled:opacity-50"
        />

        {!isRecording ? (
          hasBriefing === false ? (
            <Link
              href={`/briefings/${company}`}
              className="rounded-lg bg-amber-500 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-amber-600"
            >
              Generate Briefing First &rarr;
            </Link>
          ) : (
            <button
              onClick={handleStart}
              disabled={!canStart}
              className="rounded-lg bg-red-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Start Recording
            </button>
          )
        ) : (
          <button
            onClick={handleStop}
            className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800"
          >
            Stop &amp; Save
          </button>
        )}
      </header>

      {/* Status bar */}
      <div className="flex items-center gap-2 border-b border-stone-200 bg-stone-50 px-6 py-2 text-xs">
        <span className="font-medium text-stone-600">Status:</span>
        <span className="text-stone-500">{status}</span>
        {savedPath && !trashed && (
          <>
            <span className="ml-3 font-medium text-emerald-600">
              Saved: {savedPath.split("/").pop()}
            </span>
            <button
              onClick={async () => {
                try {
                  await interviewApi.trashTranscript(savedPath);
                  setTrashed(true);
                  setStatus("Transcript moved to Trash");
                } catch {
                  setStatus("Failed to trash transcript");
                }
              }}
              title="Move transcript to Trash"
              className="ml-2 rounded px-2 py-0.5 text-stone-400 transition-colors hover:bg-red-50 hover:text-red-500"
            >
              🗑 Trash
            </button>
          </>
        )}
        {trashed && (
          <span className="ml-3 text-stone-400 line-through">
            {savedPath?.split("/").pop()}
          </span>
        )}
      </div>

      {/* No briefing warning */}
      {hasBriefing === false && (
        <div className="border-b border-amber-200 bg-amber-50 px-6 py-3 text-sm text-amber-800">
          <span className="font-semibold text-amber-900">No briefing found for {company}. </span>
          Tribbie needs an Aglaea briefing to generate meaningful follow-up suggestions.{" "}
          <Link href={`/briefings/${company}`} className="font-medium underline hover:text-amber-900">
            Generate a briefing first &rarr;
          </Link>
        </div>
      )}

      {/* BlackHole setup guide */}
      {hasBlackhole === false && (
        <div className="border-b border-amber-200 bg-amber-50 px-6 py-3 text-sm">
          <p className="font-semibold text-amber-900">BlackHole audio device not detected</p>
          <ol className="mt-1.5 list-decimal pl-4 space-y-0.5 text-amber-800">
            <li>
              Run: <code className="rounded bg-amber-100 px-1 font-mono text-xs">brew install blackhole-2ch</code>
            </li>
            <li>Reboot your Mac</li>
            <li>
              Open <strong>Audio MIDI Setup</strong> → click{" "}
              <strong>+</strong> → <strong>Create Multi-Output Device</strong>
            </li>
            <li>Check both <strong>BlackHole 2ch</strong> and your speakers/headphones</li>
            <li>
              Set this Multi-Output Device as your system output in{" "}
              <strong>System Settings → Sound</strong>
            </li>
          </ol>
          <p className="mt-2 text-xs text-amber-700">
            Then refresh this page — the Start Recording button will become active.
          </p>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="border-b border-red-200 bg-red-50 px-6 py-3 text-sm text-red-700">
          <span className="font-medium">Error: </span>
          {error}
        </div>
      )}

      {/* Two-panel layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel: Live Transcript */}
        <div className="flex flex-1 flex-col border-r border-stone-200">
          <div className="flex items-center justify-between border-b border-stone-100 bg-stone-50 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">
              Live Transcript
            </span>
            <span className="text-xs text-stone-400">{transcript.length} segments</span>
          </div>
          <div ref={transcriptRef} className="flex-1 space-y-3 overflow-y-auto p-4">
            {transcript.length === 0 ? (
              <p className="text-sm text-stone-400">
                {isRecording
                  ? "Listening for speech..."
                  : "Transcript will appear here once recording starts."}
              </p>
            ) : (
              transcript.map((seg, i) => (
                <div key={i} className="flex gap-3">
                  <span className="mt-0.5 shrink-0 font-mono text-xs text-stone-400">
                    {new Date(seg.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })}
                  </span>
                  <p className="text-sm leading-relaxed text-stone-800">{seg.text}</p>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right column: Briefing (top) + Suggestions (bottom) — always both visible */}
        <div className="flex w-96 flex-col xl:w-[420px]">

          {/* ── Briefing panel ─────────────────────────────────────── */}
          <div className="flex h-1/2 flex-col border-b border-stone-200">
            <div className="flex shrink-0 items-center justify-between border-b border-stone-100 bg-stone-50 px-4 py-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Briefing
              </span>
              <div className="flex items-center gap-3">
                {hasStrategyHtml && (
                  <a
                    href={`${API_BASE}/api/strategy/${company}/view`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-indigo-500 transition-colors hover:text-indigo-700"
                  >
                    Strategy ↗
                  </a>
                )}
                {!hasStrategyHtml && (
                  <button
                    onClick={handleGenerateStrategy}
                    disabled={strategyGenerating}
                    title={strategyGenerating ? "Generating…" : "Generate content strategy"}
                    className="text-xs text-stone-400 transition-colors hover:text-stone-600 disabled:opacity-40"
                  >
                    {strategyGenerating ? "Strategy…" : "Strategy ↗"}
                  </button>
                )}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {briefingContent ? (
                <div className="prose prose-sm prose-stone max-w-none text-xs [&_h1]:text-sm [&_h2]:text-xs [&_h3]:text-xs [&_li]:my-0.5 [&_p]:my-1">
                  <ReactMarkdown>{briefingContent}</ReactMarkdown>
                </div>
              ) : hasBriefing === false ? (
                <p className="text-sm text-stone-400">
                  No briefing yet.{" "}
                  <Link href={`/briefings/${company}`} className="text-indigo-500 underline hover:text-indigo-700">
                    Generate one first &rarr;
                  </Link>
                </p>
              ) : (
                <p className="text-sm text-stone-400">Loading briefing…</p>
              )}
            </div>
          </div>

          {/* ── Suggestions panel ──────────────────────────────────── */}
          <div className="flex flex-1 flex-col overflow-hidden">
            <div className="flex shrink-0 items-center justify-between border-b border-stone-100 bg-stone-50 px-4 py-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Suggestions
              </span>
              {suggestions.length > 0 && (
                <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-semibold text-indigo-600">
                  {suggestions.length}
                </span>
              )}
            </div>
            <div className="flex-1 space-y-2 overflow-y-auto p-3">
              {suggestions.length === 0 ? (
                <p className="p-1 text-sm text-stone-400">
                  {isRecording
                    ? "Suggestions appear after substantial speech is detected."
                    : "Suggestions will appear here during recording."}
                </p>
              ) : (
                suggestions.map((s, i) => (
                  <div
                    key={i}
                    className={`rounded-lg border p-3 text-sm leading-relaxed ${
                      i === 0
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 shadow-sm"
                        : "border-stone-200 bg-white text-stone-700"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span className="mt-0.5 shrink-0 font-bold text-stone-400">
                        {i === 0 ? "›" : "·"}
                      </span>
                      <span className="whitespace-pre-line">{s.text}</span>
                    </div>
                    <p className="mt-1.5 text-xs text-stone-400">
                      {new Date(s.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
