"use client";

import { useParams } from "next/navigation";
import { useState, useRef, useEffect, useCallback } from "react";
import * as Popover from "@radix-ui/react-popover";
import {
  addMonths,
  eachDayOfInterval,
  endOfMonth,
  endOfWeek,
  format,
  isSameDay,
  isSameMonth,
  startOfMonth,
  startOfWeek,
} from "date-fns";
import { Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import { ghostwriterApi, postsApi, imagesApi, type Comment } from "@/lib/api";
import Link from "next/link";

interface TerminalLine {
  type: string;
  text: string;
  timestamp: number;
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
  const [activeTab, setActiveTab] = useState<"terminal" | "history" | "posts">("terminal");
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [selectedRun, setSelectedRun] = useState<RunEntry | null>(null);
  const [runEvents, setRunEvents] = useState<RunEvent[]>([]);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [posts, setPosts] = useState<any[]>([]);
  const [loadingPosts, setLoadingPosts] = useState(false);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  // LinkedIn handle is now read-only from Jacquard's users.linkedin_url.
  // The local-edit inputs/savers were removed — source of truth lives
  // server-side so the two stores can't drift.
  const [linkedinUsername, setLinkedinUsername] = useState<string | null | undefined>(undefined);
  const terminalRef = useRef<HTMLDivElement>(null);

  // Story mode: show Amphoreus story lines instead of raw tool calls.
  // We start with production-safe defaults on BOTH server and client
  // (storyMode=true) to avoid a hydration mismatch, then reconcile
  // with the real preference on mount.
  //
  // Preference precedence (mount-time):
  //   1. localStorage ``amphoreus.storyMode`` if set — explicit user
  //      choice, persists across reloads and companies.
  //   2. Otherwise: storyMode=true in prod, storyMode=false on
  //      localhost (devs default to the debug view). The toggle
  //      button is available everywhere, so anyone can flip and
  //      the choice sticks.
  const _STORY_MODE_STORAGE_KEY = "amphoreus.storyMode";
  const [storyMode, setStoryMode] = useState(true);
  const [storyLines, setStoryLines] = useState<string[]>([]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(_STORY_MODE_STORAGE_KEY);
    if (stored === "true" || stored === "false") {
      setStoryMode(stored === "true");
      return;
    }
    // No explicit preference — fall back to host-based default.
    const host = window.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      setStoryMode(false);
    }
  }, []);

  const toggleStoryMode = () => {
    setStoryMode((v) => {
      const next = !v;
      if (typeof window !== "undefined") {
        window.localStorage.setItem(_STORY_MODE_STORAGE_KEY, String(next));
      }
      return next;
    });
  };

  useEffect(() => {
    fetch("/amphoreus_story.json")
      .then((r) => r.json())
      .then((raw: string[]) => {
        // Pre-process: strip decorative separators, drop blanks, then
        // split paragraphs into individual sentences so each SSE event
        // reveals one sentence at a time.
        const sentences: string[] = [];
        for (const line of raw) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (/^[═─]{3,}$/.test(trimmed)) continue; // decorative separator
          // Book titles / chapter headers (all caps, no sentence-ending
          // punctuation) stay as one unit.
          const isHeader = /^(BOOK |EPILOGUE|THE AMPHOREUS|As remembered|End of)/i.test(trimmed)
            || trimmed === trimmed.toUpperCase();
          if (isHeader) {
            sentences.push(trimmed);
            continue;
          }
          // Split on sentence-ending punctuation followed by a space.
          // Keep the punctuation with the preceding sentence.
          const parts = trimmed.match(/[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$/g);
          if (parts) {
            for (const p of parts) {
              const s = p.trim();
              if (s) sentences.push(s);
            }
          } else {
            sentences.push(trimmed);
          }
        }
        setStoryLines(sentences);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines, runEvents]);

  useEffect(() => {
    ghostwriterApi
      .getLinkedInUsername(company)
      .then((res) => setLinkedinUsername(res.username))
      .catch(() => setLinkedinUsername(null));
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
      const res = await postsApi.list(company, 200);
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
    if (activeTab === "history") { loadRuns(); setSelectedRun(null); }
    if (activeTab === "posts") loadPosts();
  }, [activeTab, loadRuns, loadPosts]);

  // localStorage key for tracking a live Stelle run per client — lets the
  // user navigate away (or close + reopen) and rejoin the live terminal.
  const activeJobKey = `amphoreus_stelle_job_${company}`;

  // Polling-based event consumer. Replaces the old SSE iterator —
  // SSE was hanging silently for ~15-30s after Generate click (proxy
  // buffering / first-event delay), which looked indistinguishable
  // from "broken" to the operator. Refresh worked because it hit the
  // REST endpoint, which is exactly what polling does on a timer now.
  //
  // Implementation notes:
  //   - Dedupes by run_events.id (stored in the `id` query param on
  //     subsequent polls as `after_id` — a hint only; the client also
  //     tracks a Set of seen ids so reconnects / double-fires don't
  //     render dupes).
  //   - 2s poll interval → ≤2s event latency. Imperceptible over
  //     multi-minute Stelle runs.
  //   - Stops when the run reaches a terminal status OR when the
  //     caller flips ``cancelledRef.current``.
  const consumeStream = useCallback(
    async (
      jobId: string,
      afterId = 0,
      cancelledRef?: { current: boolean },
    ) => {
      const seen = new Set<number>();
      let cursor = afterId;
      const POLL_MS = 2_000;

      const applyEvents = (events: Array<{ id: number; event_type: string; data: unknown; timestamp: number }>) => {
        const fresh = events.filter((ev) => !seen.has(ev.id) && ev.id > cursor);
        if (fresh.length === 0) return;
        const toAppend: TerminalLine[] = fresh.map((ev) => {
          seen.add(ev.id);
          if (ev.id > cursor) cursor = ev.id;
          const parsed =
            typeof ev.data === "string"
              ? (() => {
                  try {
                    return JSON.parse(ev.data as string);
                  } catch {
                    return {};
                  }
                })()
              : ev.data || {};
          return {
            type: ev.event_type,
            text: extractText({ type: ev.event_type, data: parsed }),
            timestamp: ev.timestamp * 1000,
          };
        });
        setLines((prev) => [...prev, ...toAppend]);
      };

      while (true) {
        if (cancelledRef?.current) return;
        try {
          const res = await ghostwriterApi.getRunEvents(jobId);
          applyEvents(res.events || []);
          const status = res.run?.status;
          if (status === "completed" || status === "failed") {
            setLines((prev) => [
              ...prev,
              {
                type: status === "failed" ? "error" : "done",
                text: status === "failed"
                  ? `Run ${status}: ${res.run?.error || "(no error detail)"}`
                  : "Run completed.",
                timestamp: Date.now(),
              },
            ]);
            localStorage.removeItem(activeJobKey);
            setIsGenerating(false);
            return;
          }
        } catch (e) {
          // Transient fetch error — keep polling, but surface after
          // repeated failures so the operator knows something's wrong.
          // For now just log silently; polling will self-heal on the
          // next tick.
        }
        await new Promise((r) => setTimeout(r, POLL_MS));
      }
    },
    [activeJobKey],
  );

  async function handleGenerate() {
    if (isGenerating) return;
    setIsGenerating(true);
    setLines([]);
    setActiveTab("terminal");

    try {
      const { job_id } = await ghostwriterApi.generate(
        company,
        prompt || undefined,
      );
      localStorage.setItem(activeJobKey, job_id);
      setLines((prev) => [
        ...prev,
        { type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() },
      ]);
      await consumeStream(job_id, 0);
    } catch (e) {
      setLines((prev) => [
        ...prev,
        { type: "error", text: String(e), timestamp: Date.now() },
      ]);
      setIsGenerating(false);
    }
  }

  // On mount: rehydrate the terminal with the most recent run's events
  // so a page refresh preserves what the user was looking at.
  //
  // Behavior:
  //   - If a job is in-flight (localStorage key present OR the latest
  //     run is status=running): load history, then resume the live SSE
  //     stream from the last event id.
  //   - If the latest run is completed/failed: load history read-only;
  //     no stream resume.
  //   - If the client has never had a run: terminal stays empty.
  //
  // Agent filter (added 2026-04-26): only Stelle-family runs ("stelle",
  // "stelle-inline-edit") get rehydrated here. The backend's
  // ``local_runs`` table is shared across pipelines — Tribbie (interview)
  // and Cyrene (strategic review) write into the same table keyed by
  // company. Without this guard, the ghostwriter terminal would replay
  // foreign-agent events; e.g. a Tribbie session that errored on a
  // missing Cyrene brief would surface "No Cyrene brief for <slug>" in
  // the ghostwriter chronicles, even though Stelle has no such
  // dependency. Trimble/Heather hit this exact failure mode on
  // 2026-04-26.
  //
  // Previously we cleared the terminal entirely on refresh for any
  // completed/failed run, which lost the Irontomb reactions and
  // submit_draft confirmations the operator wanted to review.
  useEffect(() => {
    let cancelled = false;
    const isStelleAgent = (a: unknown): boolean =>
      typeof a === "string" && a.startsWith("stelle");

    (async () => {
      // Pick the run to rehydrate: saved "in-flight" job wins, else the
      // most recent STELLE run for this client. We fetch a small batch
      // for the fallback (instead of ``limit=1``) so a recent Tribbie
      // or Cyrene run doesn't shadow the latest Stelle one.
      const savedJobId = localStorage.getItem(activeJobKey);
      let runId: string | null = savedJobId;
      if (!runId) {
        try {
          const listing = await ghostwriterApi.getRuns(company, 10);
          const latestStelle = (listing.runs ?? []).find((r) =>
            isStelleAgent(r?.agent),
          );
          runId = latestStelle?.id ?? null;
        } catch {
          return;
        }
      }
      if (!runId || cancelled) return;

      let res;
      try {
        res = await ghostwriterApi.getRunEvents(runId);
      } catch {
        localStorage.removeItem(activeJobKey);
        return;
      }
      if (cancelled) return;

      // Drop stale localStorage that points at a non-Stelle run. This is
      // the actual fix for the Trimble/Heather symptom — a saved job_id
      // from a prior Tribbie session would otherwise replay its events
      // (including MissingCyreneBriefError) every time the user loaded
      // /ghostwriter/<slug>. Bail without rendering anything; the
      // terminal stays empty until the user clicks Generate.
      if (!isStelleAgent(res.run?.agent)) {
        if (savedJobId === runId) {
          localStorage.removeItem(activeJobKey);
        }
        return;
      }

      const status = res.run?.status;
      const isLive = status !== "completed" && status !== "failed";

      const hist: TerminalLine[] = res.events.map((ev) => {
        const parsed =
          typeof ev.data === "string"
            ? (() => {
                try {
                  return JSON.parse(ev.data as string);
                } catch {
                  return {};
                }
              })()
            : ev.data || {};
        return {
          type: ev.event_type,
          text: extractText({ type: ev.event_type, data: parsed }),
          timestamp: ev.timestamp * 1000,
        };
      });

      // Headline reflects why we're showing history — "Rejoining…" on
      // live runs (stream will append), "Last run (…)" on ended runs.
      // Race guard — the async fetches above can take 100s of ms. If
      // the user clicked Generate in that window, ``handleGenerate`` has
      // already cleared ``lines`` and pushed "Job started", AND stored
      // the NEW job id in localStorage. We must NOT clobber their state
      // or resume a stale stream for the (old) run we just fetched.
      const currentSavedId = localStorage.getItem(activeJobKey);
      const raceLost = currentSavedId !== null && currentSavedId !== runId;

      const headline = isLive
        ? `Rejoining job ${runId}…`
        : `Last run (${status}) — ${runId}`;

      // setLines via updater so we can observe the current state and
      // bail if the user has already populated it. Keep the activeTab
      // switch unconditional — they're going to the terminal either way.
      let alreadyInteracted = false;
      setLines((prev) => {
        if (prev.length > 0) {
          alreadyInteracted = true;
          return prev;
        }
        return [
          { type: "status", text: headline, timestamp: Date.now() },
          ...hist,
        ];
      });
      setActiveTab("terminal");

      if (raceLost || alreadyInteracted) {
        // User started something; their consumeStream is authoritative.
        return;
      }

      if (isLive) {
        setIsGenerating(true);
        const maxId = res.events.reduce((m, e) => (e.id > m ? e.id : m), 0);
        await consumeStream(runId, maxId);
      } else {
        // Only drop the stale key if it's the one we resolved against.
        if (currentSavedId === runId) {
          localStorage.removeItem(activeJobKey);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeJobKey, company, consumeStream]);

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center gap-4 border-b border-stone-200 bg-white px-6 py-3">
        <Link href="/home" className="text-sm text-stone-400 hover:text-stone-600">&larr;</Link>
        <h1 className="text-lg font-semibold">Stelle</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>
        <Link
          href={`/ghostwriter/${company}/calendar`}
          className="rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 transition-colors hover:border-stone-300 hover:text-stone-900"
        >
          Calendar &rarr;
        </Link>
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
          disabled={isGenerating || !linkedinUsername}
          title={!linkedinUsername ? "Set the LinkedIn username below before generating" : undefined}
          className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-50"
        >
          {isGenerating ? "Generating..." : "Generate"}
        </button>
        <button
          onClick={toggleStoryMode}
          className="rounded-lg border border-stone-300 px-3 py-1.5 text-xs text-stone-500 transition-colors hover:bg-stone-100"
          title={
            storyMode
              ? "Switch to debug view — show the real tool calls + model output"
              : "Switch to story mode — show the Amphoreus narrative overlay"
          }
        >
          {storyMode ? "Debug" : "Story"}
        </button>
      </header>

      {/* LinkedIn handle display — read-only.
          The canonical source is now Jacquard's users.linkedin_url;
          the in-app editor was retired to prevent the two-stores-drift
          bug that caused Maxwell-as-Flora's-FOC. If the handle is
          wrong, fix it in Jacquard.
       */}
      {linkedinUsername === null && (
        <div className="flex items-center gap-3 border-b border-amber-200 bg-amber-50 px-6 py-2.5 text-sm text-amber-800">
          <span className="font-semibold">No LinkedIn handle on file for this FOC</span>
          <span className="text-amber-700">
            — set <code className="rounded bg-amber-100 px-1 font-mono text-[11px]">users.linkedin_url</code> in Jacquard so Stelle can pull voice + dedup history.
          </span>
        </div>
      )}
      {linkedinUsername && (
        <div className="flex items-center gap-3 border-b border-stone-100 bg-stone-50 px-6 py-2 text-xs text-stone-500">
          <span className="text-stone-400">LinkedIn:</span>
          <a
            href={`https://linkedin.com/in/${linkedinUsername}`}
            target="_blank"
            rel="noreferrer"
            className="font-medium text-stone-700 underline decoration-stone-300 underline-offset-2 hover:decoration-stone-600"
            title="Open LinkedIn profile in new tab"
          >
            {linkedinUsername}
          </a>
          <span className="ml-auto text-[11px] text-stone-400">
            Source: Jacquard <code className="rounded bg-stone-200 px-1 font-mono">users.linkedin_url</code>. Edit there if wrong.
          </span>
        </div>
      )}

      <div className="flex border-b border-stone-200 bg-stone-50 px-6">
        {(["terminal", "posts", "history"] as const).map((tab) => (
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
            {(() => {
              // In story mode, map non-terminal events (anything
              // that isn't done/error) to story lines by position.
              // done/error always render as themselves so the user
              // sees completion regardless of mode.
              let storyIdx = 0;
              return lines.map((line, i) => {
                const isTerminal = line.type === "done" || line.type === "error";
                if (storyMode && storyLines.length > 0 && !isTerminal) {
                  if (storyIdx >= storyLines.length) return null;
                  const storyText = storyLines[storyIdx];
                  storyIdx += 1;
                  return (
                    <div key={i} className={`${getLineColor("story")} font-mono text-sm leading-loose py-1`}>
                      {storyText}
                    </div>
                  );
                }
                return (
                  <div key={i} className={getLineColor(line.type)}>
                    <span className="mr-2 text-stone-600">
                      {new Date(line.timestamp).toLocaleTimeString()}
                    </span>
                    <span className="mr-2 text-stone-500">[{line.type}]</span>
                    <span className="whitespace-pre-wrap">{line.text}</span>
                  </div>
                );
              });
            })()}
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
          <div>
            <PostsManager
              company={company}
              posts={posts}
              loading={loadingPosts}
              actionInProgress={actionInProgress}
              onAction={setActionInProgress}
              onRefresh={loadPosts}
            />
          </div>
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
    case "story": return "text-amber-200/90";
    default: return "text-stone-400";
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

/** Default picker: 7 days ahead at 09:00 local (datetime-local string). */
function defaultPublishDatetimeLocal(): string {
  const d = new Date();
  d.setDate(d.getDate() + 7);
  d.setHours(9, 0, 0, 0);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

function splitDatetimeLocal(s: string): { date: string; time: string } {
  if (s && s.length >= 16) {
    return { date: s.slice(0, 10), time: s.slice(11, 16) };
  }
  const d = defaultPublishDatetimeLocal();
  return { date: d.slice(0, 10), time: d.slice(11, 16) };
}

function mergeDatetimeLocal(dateYmd: string, timeHm: string): string {
  return `${dateYmd}T${timeHm}`;
}

function parseLocalYmd(ymd: string): Date {
  const [y, m, d] = ymd.split("-").map(Number);
  return new Date(y, (m || 1) - 1, d || 1);
}

/** Calendar popover + time input; keeps value as `YYYY-MM-DDTHH:mm` (local). */
function PublishSchedulePicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const { date: dateYmd, time: timeHm } = splitDatetimeLocal(value);
  const selectedDate = parseLocalYmd(dateYmd);
  const [open, setOpen] = useState(false);
  const [viewMonth, setViewMonth] = useState(() => startOfMonth(selectedDate));

  useEffect(() => {
    if (open) {
      setViewMonth(startOfMonth(parseLocalYmd(splitDatetimeLocal(value).date)));
    }
  }, [open, value]);

  const monthStart = startOfMonth(viewMonth);
  const monthEnd = endOfMonth(viewMonth);
  const calStart = startOfWeek(monthStart, { weekStartsOn: 0 });
  const calEnd = endOfWeek(monthEnd, { weekStartsOn: 0 });
  const days = eachDayOfInterval({ start: calStart, end: calEnd });
  const weekDays = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-end gap-2">
        <Popover.Root open={open} onOpenChange={setOpen}>
          <Popover.Trigger asChild>
            <button
              type="button"
              className="mt-1 flex min-h-[2.25rem] min-w-[10.5rem] items-center justify-between gap-2 rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-left text-sm text-stone-200 hover:border-stone-500"
            >
              <span className="flex items-center gap-2">
                <Calendar className="h-4 w-4 shrink-0 text-stone-400" aria-hidden />
                {format(selectedDate, "MMM d, yyyy")}
              </span>
            </button>
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Content
              className="z-[60] w-[min(100vw-2rem,18rem)] rounded-lg border border-stone-600 bg-stone-900 p-3 shadow-xl"
              sideOffset={4}
              align="start"
            >
              <div className="mb-2 flex items-center justify-between gap-1">
                <button
                  type="button"
                  aria-label="Previous month"
                  className="rounded p-1 text-stone-400 hover:bg-stone-800 hover:text-stone-200"
                  onClick={() => setViewMonth((m) => addMonths(m, -1))}
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <span className="text-xs font-medium text-stone-200">
                  {format(viewMonth, "MMMM yyyy")}
                </span>
                <button
                  type="button"
                  aria-label="Next month"
                  className="rounded p-1 text-stone-400 hover:bg-stone-800 hover:text-stone-200"
                  onClick={() => setViewMonth((m) => addMonths(m, 1))}
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
              <div className="mb-1 grid grid-cols-7 gap-0.5 text-center text-[10px] font-medium uppercase tracking-wide text-stone-500">
                {weekDays.map((d) => (
                  <div key={d}>{d}</div>
                ))}
              </div>
              <div className="grid grid-cols-7 gap-0.5">
                {days.map((day) => {
                  const inMonth = isSameMonth(day, viewMonth);
                  const isSelected = isSameDay(day, selectedDate);
                  const ymd = format(day, "yyyy-MM-dd");
                  return (
                    <button
                      key={ymd}
                      type="button"
                      onClick={() => {
                        onChange(mergeDatetimeLocal(ymd, timeHm));
                        setViewMonth(startOfMonth(day));
                        setOpen(false);
                      }}
                      className={
                        "flex h-8 items-center justify-center rounded text-xs tabular-nums " +
                        (!inMonth
                          ? "text-stone-600 hover:bg-stone-800/80 hover:text-stone-400"
                          : isSelected
                            ? "bg-cyan-800 font-medium text-white"
                            : "text-stone-300 hover:bg-stone-800")
                      }
                    >
                      {format(day, "d")}
                    </button>
                  );
                })}
              </div>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
        <label className="block text-xs font-medium text-stone-400">
          Time (local)
          <input
            type="time"
            value={timeHm}
            onChange={(e) => onChange(mergeDatetimeLocal(dateYmd, e.target.value))}
            className="mt-1 block w-full min-w-[7rem] rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-sm text-stone-200"
          />
        </label>
      </div>
      <p className="text-[11px] text-stone-500">
        Selected:{" "}
        <time dateTime={value || mergeDatetimeLocal(dateYmd, timeHm)}>
          {format(parseLocalYmd(dateYmd), "EEEE, MMMM d, yyyy")} at {timeHm}
        </time>
      </p>
    </div>
  );
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
  const [citationData, setCitationData] = useState<Record<string, string[]>>({});
  // Tracks which post just had its body copied — shows a transient
  // "Copied!" affordance on the Copy button. Reset after ~1.5s.
  const [copiedPostId, setCopiedPostId] = useState<string | null>(null);

  // Copy a post body to the clipboard. Uses the modern
  // navigator.clipboard API; falls back to a hidden-textarea +
  // execCommand for non-secure contexts or browsers that block
  // clipboard access. Last resort: alert the user to copy manually.
  async function handleCopyPost(postId: string, content: string) {
    if (!content) return;
    try {
      await navigator.clipboard.writeText(content);
      setCopiedPostId(postId);
      window.setTimeout(() => {
        setCopiedPostId((cur) => (cur === postId ? null : cur));
      }, 1500);
      return;
    } catch {
      /* fall through to legacy path */
    }
    try {
      const ta = document.createElement("textarea");
      ta.value = content;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopiedPostId(postId);
      window.setTimeout(() => {
        setCopiedPostId((cur) => (cur === postId ? null : cur));
      }, 1500);
    } catch {
      window.alert(
        "Couldn't copy automatically — please select the post body and copy with Cmd/Ctrl+C.",
      );
    }
  }

  // Comments (draft_feedback) state. Per-post so toggling one panel
  // doesn't clobber another's loaded list.
  const [commentsByPost, setCommentsByPost] = useState<Record<string, Comment[]>>({});
  const [openCommentsFor, setOpenCommentsFor] = useState<string | null>(null);
  const [composerText, setComposerText] = useState<Record<string, string>>({});
  // Inline mode — populated by the selection handler inside the post
  // (2026-04-23) Removed ``composerSelection`` + ``captureInlineSelection``.
  // The composer textarea below the body is now post-wide-ONLY — the
  // old onMouseUp auto-staging of a highlighted range into the
  // composer was confusing (a "Commenting on selection…" blurb would
  // appear without the operator having opted in). Inline comments are
  // created exclusively through the right-click popup
  // (``openInlineCommentPopup``), which is explicit and matches the
  // mental model operators expect from any other text editor.
  // Inline-edit-in-place state for existing comments.
  const [editingCommentId, setEditingCommentId] = useState<string | null>(null);
  const [editingCommentText, setEditingCommentText] = useState("");
  // Soft cross-highlight: hover/click a mark or its rail card →
  // briefly highlight the matching pair so the operator can see the
  // anchor relationship at a glance.
  const [highlightedCommentId, setHighlightedCommentId] = useState<string | null>(null);
  const [imageJobId, setImageJobId] = useState<string | null>(null);
  const [imageLines, setImageLines] = useState<{ type: string; text: string }[]>([]);
  const [generatingImageFor, setGeneratingImageFor] = useState<string | null>(null);
  const [pendingImageByPost, setPendingImageByPost] = useState<Record<string, string>>({});
  const [imageFeedbackByPost, setImageFeedbackByPost] = useState<Record<string, string>>({});
  const [pushModalPost, setPushModalPost] = useState<any | null>(null);
  const [pushAllOpen, setPushAllOpen] = useState(false);
  const [pushAllPostsPerMonth, setPushAllPostsPerMonth] = useState<8 | 12>(12);
  const [pushPublishLocal, setPushPublishLocal] = useState("");
  const [pushApproverIds, setPushApproverIds] = useState<Set<string>>(new Set());
  const [pushApprovalsBlocking, setPushApprovalsBlocking] = useState(true);
  const [ordinalUsers, setOrdinalUsers] = useState<any[]>([]);
  const [loadingOrdinalUsers, setLoadingOrdinalUsers] = useState(false);
  const [ordinalUsersError, setOrdinalUsersError] = useState<string | null>(null);

  useEffect(() => {
    if (!pushModalPost && !pushAllOpen) return;
    setOrdinalUsersError(null);
    setLoadingOrdinalUsers(true);
    ghostwriterApi
      .getOrdinalUsers(company)
      .then((res) => setOrdinalUsers(res.users || []))
      .catch(() => {
        setOrdinalUsers([]);
        setOrdinalUsersError("Could not load Ordinal workspace users (check API key in ordinal_auth_rows).");
      })
      .finally(() => setLoadingOrdinalUsers(false));
  }, [pushModalPost, pushAllOpen, company]);

  // Prefetch comments so each post card can show a "N comments" badge
  // without the operator having to open the panel. Parallel fetch, silent
  // on failure — per-post panels still refetch on toggle/mutation, so a
  // prefetch miss just means the badge is blank until then.
  useEffect(() => {
    const ids = posts.map((p) => p.id).filter(Boolean);
    if (ids.length === 0) return;
    let cancelled = false;
    Promise.all(
      ids.map((id) =>
        postsApi.listComments(id, true)
          .then((res) => [id, res.comments ?? []] as const)
          .catch(() => [id, [] as Comment[]] as const),
      ),
    ).then((pairs) => {
      if (cancelled) return;
      setCommentsByPost((cur) => {
        const next = { ...cur };
        for (const [id, list] of pairs) next[id] = list;
        return next;
      });
    });
    return () => { cancelled = true; };
  }, [posts]);

  function openPushModal(post: any) {
    setPushPublishLocal(defaultPublishDatetimeLocal());
    setPushApproverIds(new Set());
    setPushApprovalsBlocking(true);
    setPushModalPost(post);
  }

  function openPushAllModal() {
    setPushApproverIds(new Set());
    setPushApprovalsBlocking(true);
    setPushAllPostsPerMonth(12);
    setPushAllOpen(true);
  }

  async function handleConfirmPushAll() {
    onAction("__push_all__");
    try {
      const approvals = Array.from(pushApproverIds).map((userId) => ({
        userId,
        isBlocking: pushApprovalsBlocking,
      }));
      const res = await postsApi.pushAll(company, pushAllPostsPerMonth, { approvals });
      const errLines = res.errors?.length ? `\n\nErrors:\n${res.errors.slice(0, 8).join("\n")}` : "";
      if (res.success) {
        window.alert(
          `Pushed ${res.pushed} of ${res.total} draft(s) on ${res.cadence} cadence (UTC 09:00 per slot).` +
            (res.first_url ? `\n\nOpen: ${res.first_url}` : "") +
            errLines
        );
        setPushAllOpen(false);
        onRefresh();
      } else {
        window.alert(`Push-all failed (${res.pushed}/${res.total}).${errLines}`);
      }
    } catch (e) {
      window.alert(`Push-all failed: ${e}`);
    } finally {
      onAction(null);
    }
  }

  function toggleApprover(id: string) {
    setPushApproverIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleGenerateImage(
    post: any,
    options?: { feedback?: string; referenceImageId?: string }
  ) {
    const pid = post.id;
    setGeneratingImageFor(pid);
    setImageLines([]);
    setImageJobId(null);
    try {
      const { job_id } = await imagesApi.generate(company, post.content, undefined, {
        feedback: options?.feedback,
        referenceImageId: options?.referenceImageId,
        localPostId: pid,
      });
      setImageJobId(job_id);
      for await (const data of imagesApi.streamJob(job_id)) {
        const d = (data.data || {}) as { image_id?: string; text?: string; message?: string; name?: string; output?: string };
        const text =
          d.text || d.message || d.name || d.output?.slice?.(0, 200) || JSON.stringify(d);
        setImageLines((prev) => [...prev, { type: data.type, text }]);
        if (data.type === "done" && d.image_id) {
          setPendingImageByPost((prev) => ({ ...prev, [pid]: d.image_id as string }));
        }
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

  async function handleApproveLinkedImage(post: any) {
    const img = pendingImageByPost[post.id] || post.linked_image_id;
    if (!img) {
      window.alert("Generate an image first, then approve it for Ordinal push.");
      return;
    }
    onAction(post.id);
    try {
      await postsApi.update(post.id, company, { linked_image_id: img });
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  async function handleRegenerateWithFeedback(post: any) {
    const ref = pendingImageByPost[post.id] || post.linked_image_id;
    const fb = (imageFeedbackByPost[post.id] || "").trim();
    if (!fb && !ref) {
      window.alert("Generate an image first, or add revision notes (optionally with an existing image as reference).");
      return;
    }
    await handleGenerateImage(post, {
      feedback: fb || "Improve the composite image to better match the post; apply concrete visual changes.",
      referenceImageId: ref,
    });
  }

  async function handleDelete(postId: string) {
    if (!confirm("Delete this post?")) return;
    onAction(postId);
    try {
      await postsApi.delete(postId);
      onRefresh();
    } catch (e) {
      // Surface the backend detail string instead of silently swallowing —
      // a missing catch here turned real 500s into "button did nothing."
      window.alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      onAction(null);
    }
  }

  // Reject — distinct from Delete. Preserves the draft row + all paired
  // draft_feedback comments so Stelle / Aglaea can learn from the
  // rejection on future runs. Use when the client explicitly turns
  // down a draft (Ordinal "Blocked" or equivalent).
  //
  // No separate reason prompt: comments ARE the rejection's reason.
  // Operators leave inline / post-wide comments on the draft (which
  // get persisted to draft_feedback), then click Reject. The comments
  // surface to Stelle's bundle under the REJECTED block paired to the
  // draft body. Adding a parallel rejection_reason field would be
  // redundant signal capture — same channel, two collection points.
  async function handleReject(postId: string) {
    const ok = window.confirm(
      "Reject this draft?\n\n" +
        "Any comments you've left on this draft (inline or post-wide) " +
        "will be the rejection's learning signal — Stelle and Aglaea " +
        "see them paired to this draft on future runs. Leave them first " +
        "if you want them to inform learning.\n\n" +
        "The draft is preserved (NOT deleted) and won't act as dedup " +
        "signal — future runs can write on the same topic."
    );
    if (!ok) return;
    onAction(postId);
    try {
      await postsApi.reject(postId);
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  // handleSetPublishDate was removed 2026-04-28. Pairing now happens
  // automatically on every Apify scrape via the semantic match-back
  // worker — operators don't tell the system when a draft was
  // published, the worker pairs it the next time the LinkedIn post
  // is scraped. Paired drafts surface their published-version body
  // inline in the Posts tab via the ``published_post_text`` field
  // returned by GET /api/posts (see PublishedVersion expander below).

  async function handleRewrite(post: any) {
    onAction(post.id);
    try {
      const res = await postsApi.rewrite(post.id, company, post.content);
      // 2026-05-01: rewrite is EPHEMERAL by design — backend no longer
      // persists. We get the rewritten text back, drop it into the
      // edit panel, and let the operator decide whether to Save
      // (commits via the existing Edit Save flow → update_local_post)
      // or Cancel (original stays untouched). The post body in the
      // card view does NOT change until the operator hits Save —
      // the rewrite lives only in the edit textarea until then.
      const rewritten =
        typeof res.result === "string"
          ? res.result
          : (res.result?.final_post || res.result?.content || "");
      if (rewritten) {
        setEditingId(post.id);
        setEditText(rewritten);
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
      if (res.citation_comments?.length) {
        setCitationData((prev) => ({ ...prev, [post.id]: res.citation_comments }));
      }
    } finally {
      onAction(null);
    }
  }

  async function handleSaveEdit(postId: string) {
    onAction(postId);
    try {
      await postsApi.update(postId, company, { content: editText });
      setEditingId(null);
      setEditText("");
      onRefresh();
    } finally {
      onAction(null);
    }
  }

  // --- Comments (draft_feedback) ---
  // Load comments for a single post; used on panel toggle and after
  // every mutation so counts stay accurate.
  async function refreshComments(postId: string) {
    try {
      const res = await postsApi.listComments(postId, true);
      setCommentsByPost((m) => ({ ...m, [postId]: res.comments ?? [] }));
    } catch {
      // Mirror may be pre-migration; show empty list rather than an error.
      setCommentsByPost((m) => ({ ...m, [postId]: [] }));
    }
  }

  function toggleComments(postId: string) {
    setOpenCommentsFor((cur) => {
      const next = cur === postId ? null : postId;
      if (next) void refreshComments(postId);
      return next;
    });
  }

  async function handleAddComment(postId: string) {
    // Post-wide only — inline comments go through the right-click
    // popup (``openInlineCommentPopup``). No selection to thread here.
    const body = (composerText[postId] || "").trim();
    if (!body) return;
    onAction(postId);
    try {
      await postsApi.addComment(postId, body, {
        authorEmail: null,   // backend stamps from auth when wired
        authorName: null,
      });
      setComposerText((m) => ({ ...m, [postId]: "" }));
      await refreshComments(postId);
    } finally {
      onAction(null);
    }
  }

  async function handleResolveComment(postId: string, commentId: string) {
    try {
      await postsApi.resolveComment(postId, commentId);
      await refreshComments(postId);
    } catch {}
  }

  async function handleDeleteComment(postId: string, commentId: string) {
    if (!window.confirm("Delete this comment? This cannot be undone.")) return;
    try {
      await postsApi.deleteComment(postId, commentId);
      await refreshComments(postId);
    } catch {}
  }

  function startEditingComment(c: Comment) {
    setEditingCommentId(c.id);
    setEditingCommentText(c.body);
  }

  async function handleSaveEditComment(postId: string, commentId: string) {
    const next = editingCommentText.trim();
    if (!next) return;
    try {
      await postsApi.editComment(postId, commentId, next);
      setEditingCommentId(null);
      setEditingCommentText("");
      await refreshComments(postId);
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e));
    }
  }

  // Pair-scrolling helpers used by the highlight ↔ comment-card link.
  // Scoped to a single post via the prefix so two cards on the page
  // can't collide on element ids.
  function scrollToCommentCard(postId: string, commentId: string) {
    setHighlightedCommentId(commentId);
    const el = document.getElementById(`cmt-card-${postId}-${commentId}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    window.setTimeout(() => setHighlightedCommentId(null), 1500);
  }
  function scrollToCommentMark(postId: string, commentId: string) {
    setHighlightedCommentId(commentId);
    const el = document.getElementById(`cmt-mark-${postId}-${commentId}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    window.setTimeout(() => setHighlightedCommentId(null), 1500);
  }

  // Right-click inline-comment popup. When the operator selects text
  // in a post body and right-clicks, we pop an anchored composer at
  // the cursor so they can drop a thought without scrolling to the
  // comment rail. The popup targets feel like the rest of the
  // ghostwriter page (dark surface, cyan accents, thin rings).
  //
  // Shape: ``{postId, content, x, y, selection}`` — content is the
  // full post body text (used to re-verify the offsets if the popup
  // re-renders), x/y are viewport-space pixel coordinates where the
  // card anchors, selection is the captured {start, end, text}.
  const [inlinePopup, setInlinePopup] = useState<
    | {
        postId: string;
        content: string;
        x: number;
        y: number;
        selection: { start: number; end: number; text: string };
        draft: string;
      }
    | null
  >(null);
  const [inlinePopupBusy, setInlinePopupBusy] = useState(false);
  const inlinePopupTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Click-on-mark edit/delete popup. Left-clicking an existing inline
  // highlight opens a compact card at the mark's position showing the
  // comment body with Edit + Delete affordances. Replaces the older
  // "scroll to rail" behavior — the rail still exists, but editing
  // inline without leaving the draft is the expected flow. Shares
  // theme with ``inlinePopup`` (the add-comment variant) so the two
  // paths feel like the same primitive.
  const [markPopup, setMarkPopup] = useState<
    | {
        postId: string;
        commentId: string;
        x: number;
        y: number;
        mode: "view" | "edit";
        draft: string;
      }
    | null
  >(null);
  const [markPopupBusy, setMarkPopupBusy] = useState(false);
  const markPopupTextareaRef = useRef<HTMLTextAreaElement | null>(null);

  function openMarkPopup(
    e: React.MouseEvent<HTMLElement>,
    postId: string,
    commentId: string,
    initialBody: string,
  ) {
    e.preventDefault();
    e.stopPropagation();
    setMarkPopup({
      postId,
      commentId,
      x: e.clientX,
      y: e.clientY,
      mode: "view",
      draft: initialBody,
    });
  }

  async function handleMarkPopupSaveEdit() {
    if (!markPopup) return;
    const next = markPopup.draft.trim();
    if (!next) return;
    setMarkPopupBusy(true);
    try {
      await postsApi.editComment(markPopup.postId, markPopup.commentId, next);
      await refreshComments(markPopup.postId);
      setMarkPopup(null);
    } catch (e) {
      window.alert(`Edit failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setMarkPopupBusy(false);
    }
  }

  async function handleMarkPopupDelete() {
    if (!markPopup) return;
    if (!window.confirm("Delete this inline comment? The highlighted text will lose its mark.")) return;
    setMarkPopupBusy(true);
    try {
      await postsApi.deleteComment(markPopup.postId, markPopup.commentId);
      await refreshComments(markPopup.postId);
      setMarkPopup(null);
    } catch (e) {
      window.alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setMarkPopupBusy(false);
    }
  }

  useEffect(() => {
    if (!markPopup) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setMarkPopup(null);
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && markPopup?.mode === "edit") {
        e.preventDefault();
        void handleMarkPopupSaveEdit();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [markPopup]);

  // onContextMenu handler bound to each post's rendered body. Reads
  // the current browser selection, resolves it to char offsets inside
  // ``postContent``, suppresses the native menu, and opens the popup
  // at the click coordinates. Falls through (native menu shows) when
  // there's no selection — operators can still copy/paste via right-
  // click on un-selected text.
  function openInlineCommentPopup(
    e: React.MouseEvent<HTMLElement>,
    postId: string,
    postContent: string,
  ) {
    if (typeof window === "undefined") return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;                     // no selection → native menu
    const text = sel.toString();
    if (!text.trim()) return;
    const start = postContent.indexOf(text);
    if (start < 0) return;                                    // selection spans outside body
    e.preventDefault();
    const end = start + text.length;
    setInlinePopup({
      postId,
      content: postContent,
      x: e.clientX,
      y: e.clientY,
      selection: { start, end, text },
      draft: "",
    });
    // Defer focus until the popup has mounted + react committed.
    requestAnimationFrame(() => {
      inlinePopupTextareaRef.current?.focus();
    });
  }

  async function handleInlinePopupSave() {
    if (!inlinePopup) return;
    const body = inlinePopup.draft.trim();
    if (!body) return;
    setInlinePopupBusy(true);
    try {
      await postsApi.addComment(inlinePopup.postId, body, {
        authorEmail: null,
        authorName: null,
        selection: inlinePopup.selection,
      });
      await refreshComments(inlinePopup.postId);
      // Make sure the newly-commented post's rail is open so the
      // operator sees their comment land.
      setOpenCommentsFor(inlinePopup.postId);
      setInlinePopup(null);
    } catch (e) {
      window.alert(`Could not save comment: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setInlinePopupBusy(false);
    }
  }

  // Dismiss the popup on Escape + on clicks outside its box. We don't
  // use a portal because the popup never leaves the page's scrolling
  // viewport, and anchoring to viewport coordinates is enough.
  useEffect(() => {
    if (!inlinePopup) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") { setInlinePopup(null); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inlinePopup]);

  // --- Revert to Stelle's original (pre-Castorice). ---
  async function handleRevertToOriginal(postId: string) {
    if (!window.confirm(
      "Revert this post's content to Stelle's original version (before Castorice fact-check)? " +
      "The current version will be saved in the revision history and can be restored from there later."
    )) return;
    onAction(postId);
    try {
      await postsApi.revertToOriginal(postId);
      onRefresh();
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e));
    } finally {
      onAction(null);
    }
  }

  async function handleConfirmPush() {
    if (!pushModalPost) return;
    const post = pushModalPost;
    const t = new Date(pushPublishLocal);
    if (Number.isNaN(t.getTime())) {
      window.alert("Invalid date/time.");
      return;
    }
    onAction(post.id);
    try {
      const approvals = Array.from(pushApproverIds).map((userId) => ({
        userId,
        isBlocking: pushApprovalsBlocking,
      }));
      const res = await postsApi.push(company, post.content, [], {
        postId: post.id,
        publishAt: t.toISOString(),
        approvals,
      });
      if (res.success) {
        const oid = res.ordinal_post_ids?.[0];
        window.alert(
          (res.result ? `Pushed to Ordinal.\n\nOpen: ${res.result}` : "Pushed to Ordinal.") +
            (oid ? `\n\nOrdinal post id (saved on this draft): ${oid}` : "") +
            (post.linked_image_id || pendingImageByPost[post.id]
              ? "\n\nApproved image is attached when PUBLIC_BASE_URL on the API points to a URL Ordinal can reach (see server .env)."
              : "")
        );
        setPushModalPost(null);
        onRefresh();
      } else {
        window.alert(`Push failed:\n${String(res.result || "Unknown error")}`);
      }
    } catch (e) {
      window.alert(`Push failed: ${e}`);
    } finally {
      onAction(null);
    }
  }

  if (loading) return <p className="text-stone-500">Loading posts...</p>;

  return (
    <div className="space-y-3">
      {/* Inline-comment popup. Anchored at the operator's right-click
          point in the post body; dismisses on Escape or click-outside.
          Theme: dark surface with thin cyan ring to match the rest of
          the ghostwriter page without competing for attention. The
          backdrop is transparent so the operator keeps seeing the
          highlighted text while they type. */}
      {inlinePopup && (
        <div
          className="fixed inset-0 z-[60]"
          onClick={() => { if (!inlinePopupBusy) setInlinePopup(null); }}
        >
          <div
            role="dialog"
            aria-label="Add inline comment"
            onClick={(e) => e.stopPropagation()}
            style={{
              // Keep the card inside the viewport even when the click
              // lands in the far-right/bottom corner. Clamp rather
              // than transform-translate so textarea selection stays
              // pixel-aligned for keyboard navigation.
              left: Math.min(Math.max(12, inlinePopup.x), (typeof window !== "undefined" ? window.innerWidth : 2000) - 340),
              top: Math.min(Math.max(12, inlinePopup.y + 8), (typeof window !== "undefined" ? window.innerHeight : 2000) - 220),
            }}
            className="absolute w-[22rem] rounded-lg border border-stone-700 bg-stone-900/95 p-3 shadow-2xl ring-1 ring-cyan-900/40 backdrop-blur"
          >
            <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-cyan-400">
              Inline comment
            </div>
            <div className="mb-2 rounded border border-amber-400/30 bg-amber-300/10 px-2 py-1 text-[11px] italic text-amber-200">
              “{inlinePopup.selection.text.slice(0, 140)}{inlinePopup.selection.text.length > 140 ? "…" : ""}”
            </div>
            <textarea
              ref={inlinePopupTextareaRef}
              value={inlinePopup.draft}
              onChange={(e) => setInlinePopup((p) => p ? { ...p, draft: e.target.value } : p)}
              onKeyDown={(e) => {
                // Cmd/Ctrl-Enter commits; Escape cancels (handled at window level).
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  e.preventDefault();
                  void handleInlinePopupSave();
                }
              }}
              rows={3}
              placeholder="What's the issue with this chunk?"
              disabled={inlinePopupBusy}
              className="w-full resize-none rounded border border-stone-700 bg-stone-950 px-2 py-1.5 text-sm text-stone-100 placeholder:text-stone-500 focus:border-cyan-700 focus:outline-none focus:ring-1 focus:ring-cyan-800"
            />
            <div className="mt-2 flex items-center justify-between">
              <span className="text-[10px] text-stone-500">
                {inlinePopup.draft.trim().length > 0 ? "⌘↵ to save · Esc to cancel" : "Esc to cancel"}
              </span>
              <div className="flex gap-1.5">
                <button
                  type="button"
                  onClick={() => { if (!inlinePopupBusy) setInlinePopup(null); }}
                  disabled={inlinePopupBusy}
                  className="rounded px-2 py-1 text-xs text-stone-400 hover:text-stone-200"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => void handleInlinePopupSave()}
                  disabled={inlinePopupBusy || !inlinePopup.draft.trim()}
                  className="rounded bg-cyan-950 px-2.5 py-1 text-xs font-medium text-cyan-300 ring-1 ring-cyan-800 hover:bg-cyan-900 disabled:opacity-40"
                >
                  {inlinePopupBusy ? "Saving…" : "Save"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mark-click edit/delete popup. Same theme as the add-comment
          popup above, but anchored on an existing highlight and
          showing view→edit transitions + a delete affordance. */}
      {markPopup && (
        <div
          className="fixed inset-0 z-[60]"
          onClick={() => { if (!markPopupBusy) setMarkPopup(null); }}
        >
          <div
            role="dialog"
            aria-label="Edit inline comment"
            onClick={(e) => e.stopPropagation()}
            style={{
              left: Math.min(Math.max(12, markPopup.x), (typeof window !== "undefined" ? window.innerWidth : 2000) - 340),
              top:  Math.min(Math.max(12, markPopup.y + 8), (typeof window !== "undefined" ? window.innerHeight : 2000) - 220),
            }}
            className="absolute w-[22rem] rounded-lg border border-stone-700 bg-stone-900/95 p-3 shadow-2xl ring-1 ring-amber-400/20 backdrop-blur"
          >
            <div className="mb-1.5 flex items-center justify-between">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-amber-300">
                Inline comment
              </div>
              {(() => {
                const c = (commentsByPost[markPopup.postId] ?? []).find((x) => x.id === markPopup.commentId);
                if (!c) return null;
                const author = (c.author_email || c.author_name || "").trim();
                if (!author) return null;
                return <div className="text-[10px] text-stone-500">{author}</div>;
              })()}
            </div>

            {markPopup.mode === "view" ? (
              <>
                <div className="max-h-40 overflow-y-auto whitespace-pre-wrap rounded border border-stone-700 bg-stone-950/60 px-2 py-1.5 text-xs text-stone-200">
                  {markPopup.draft || <span className="italic text-stone-500">(empty)</span>}
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => void handleMarkPopupDelete()}
                    disabled={markPopupBusy}
                    className="rounded px-2 py-1 text-xs text-red-400 hover:bg-red-950/30 hover:text-red-300 disabled:opacity-40"
                  >
                    Delete
                  </button>
                  <div className="flex gap-1.5">
                    <button
                      type="button"
                      onClick={() => { if (!markPopupBusy) setMarkPopup(null); }}
                      disabled={markPopupBusy}
                      className="rounded px-2 py-1 text-xs text-stone-400 hover:text-stone-200"
                    >
                      Close
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setMarkPopup((p) => p ? { ...p, mode: "edit" } : p);
                        requestAnimationFrame(() => markPopupTextareaRef.current?.focus());
                      }}
                      disabled={markPopupBusy}
                      className="rounded bg-cyan-950 px-2.5 py-1 text-xs font-medium text-cyan-300 ring-1 ring-cyan-800 hover:bg-cyan-900 disabled:opacity-40"
                    >
                      Edit
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <>
                <textarea
                  ref={markPopupTextareaRef}
                  value={markPopup.draft}
                  onChange={(e) => setMarkPopup((p) => p ? { ...p, draft: e.target.value } : p)}
                  rows={4}
                  disabled={markPopupBusy}
                  className="w-full resize-none rounded border border-stone-700 bg-stone-950 px-2 py-1.5 text-sm text-stone-100 focus:border-cyan-700 focus:outline-none focus:ring-1 focus:ring-cyan-800"
                />
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-[10px] text-stone-500">⌘↵ to save · Esc to cancel</span>
                  <div className="flex gap-1.5">
                    <button
                      type="button"
                      onClick={() => setMarkPopup((p) => p ? { ...p, mode: "view" } : p)}
                      disabled={markPopupBusy}
                      className="rounded px-2 py-1 text-xs text-stone-400 hover:text-stone-200"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleMarkPopupSaveEdit()}
                      disabled={markPopupBusy || !markPopup.draft.trim()}
                      className="rounded bg-cyan-950 px-2.5 py-1 text-xs font-medium text-cyan-300 ring-1 ring-cyan-800 hover:bg-cyan-900 disabled:opacity-40"
                    >
                      {markPopupBusy ? "Saving…" : "Save"}
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {pushModalPost && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div
            className="max-h-[min(90vh,36rem)] w-full max-w-lg overflow-y-auto rounded-xl border border-stone-700 bg-stone-900 p-5 shadow-xl"
            role="dialog"
            aria-labelledby="push-ordinal-title"
          >
            <h4 id="push-ordinal-title" className="mb-3 text-sm font-semibold text-stone-100">
              Push to Ordinal
            </h4>
            {(pushModalPost.linked_image_id || pendingImageByPost[pushModalPost.id]) && (
              <p className="mb-3 rounded border border-amber-900/40 bg-amber-950/20 px-2 py-1.5 text-xs text-amber-200/90">
                Image: Ordinal will attach the approved or preview PNG when{" "}
                <code className="text-amber-100/80">PUBLIC_BASE_URL</code> on the API is a URL Ordinal can fetch.
                {!pushModalPost.linked_image_id && pendingImageByPost[pushModalPost.id] && (
                  <span className="block pt-1 text-amber-300/80">
                    Save with &quot;Use for Ordinal&quot; first if you want this push to use the new preview.
                  </span>
                )}
              </p>
            )}
            <p className="mb-3 text-xs text-stone-500">
              Only <span className="text-stone-300">this draft</span> is pushed (the saved post body from the list below),
              with its stored citations and why-post as thread comments—not your Cyrene markdown file.
            </p>
            <p className="mb-3 text-xs text-stone-500">
              Schedule time is sent to Ordinal as <code className="text-stone-400">publishAt</code> (UTC).
              Optional approvers receive requests per{" "}
              <a
                className="text-cyan-500 hover:underline"
                href="https://docs.tryordinal.com/api-reference/approvals/create-approval-requests"
                target="_blank"
                rel="noreferrer"
              >
                Ordinal approvals API
              </a>
              .
            </p>
            <div className="mb-3">
              <div className="mb-1 text-xs font-medium text-stone-400">Publish date &amp; time (local)</div>
              <PublishSchedulePicker value={pushPublishLocal} onChange={setPushPublishLocal} />
            </div>
            <div className="mb-3">
              <div className="mb-1 text-xs font-medium text-stone-400">Approvers (optional)</div>
              {loadingOrdinalUsers && (
                <p className="text-xs text-stone-500">Loading workspace users…</p>
              )}
              {ordinalUsersError && (
                <p className="text-xs text-amber-400">{ordinalUsersError}</p>
              )}
              {!loadingOrdinalUsers && !ordinalUsersError && ordinalUsers.length === 0 && (
                <p className="text-xs text-stone-500">No users returned.</p>
              )}
              <ul className="max-h-40 space-y-1 overflow-y-auto rounded border border-stone-800 p-2">
                {ordinalUsers.map((u: any) => {
                  const uid = u.id || u.userId;
                  if (!uid) return null;
                  const name = [u.firstName, u.lastName].filter(Boolean).join(" ") || "User";
                  return (
                    <li key={uid}>
                      <label className="flex cursor-pointer items-center gap-2 text-xs text-stone-300">
                        <input
                          type="checkbox"
                          checked={pushApproverIds.has(uid)}
                          onChange={() => toggleApprover(uid)}
                        />
                        <span>
                          {name} {u.email ? <span className="text-stone-500">({u.email})</span> : null}
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              {pushApproverIds.size > 0 && (
                <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-stone-400">
                  <input
                    type="checkbox"
                    checked={pushApprovalsBlocking}
                    onChange={(e) => setPushApprovalsBlocking(e.target.checked)}
                  />
                  Blocking approvals
                </label>
              )}
            </div>
            <div className="flex justify-end gap-2 border-t border-stone-800 pt-3">
              <button
                type="button"
                onClick={() => setPushModalPost(null)}
                className="rounded bg-stone-800 px-3 py-1.5 text-xs text-stone-300 hover:bg-stone-700"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleConfirmPush()}
                disabled={actionInProgress === pushModalPost.id}
                className="rounded bg-cyan-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-cyan-700 disabled:opacity-50"
              >
                {actionInProgress === pushModalPost.id ? "Pushing…" : "Push"}
              </button>
            </div>
          </div>
        </div>
      )}

      {pushAllOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div
            className="max-h-[min(90vh,36rem)] w-full max-w-lg overflow-y-auto rounded-xl border border-stone-700 bg-stone-900 p-5 shadow-xl"
            role="dialog"
            aria-labelledby="push-all-ordinal-title"
          >
            <h4 id="push-all-ordinal-title" className="mb-3 text-sm font-semibold text-stone-100">
              Push all drafts to Ordinal
            </h4>
            <p className="mb-3 text-xs text-stone-500">
              Each listed draft becomes its own Ordinal post. Publish dates are the next{" "}
              <strong className="font-medium text-stone-300">Mon / Wed / Thu</strong> (12/mo) or{" "}
              <strong className="font-medium text-stone-300">Tue / Thu</strong> (8/mo) slots starting from today
              (UTC calendar day), at <code className="text-stone-400">09:00 UTC</code> per slot. Oldest drafts get
              the earliest slots.
            </p>
            <label className="mb-3 block text-xs font-medium text-stone-400">
              Posting cadence
              <select
                value={pushAllPostsPerMonth}
                onChange={(e) => setPushAllPostsPerMonth(Number(e.target.value) as 8 | 12)}
                className="mt-1 w-full rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-sm text-stone-200"
              >
                <option value={12}>12 posts / month — Mon, Wed, Thu</option>
                <option value={8}>8 posts / month — Tue, Thu</option>
              </select>
            </label>
            <div className="mb-3">
              <div className="mb-1 text-xs font-medium text-stone-400">Approvers (optional)</div>
              {loadingOrdinalUsers && (
                <p className="text-xs text-stone-500">Loading workspace users…</p>
              )}
              {ordinalUsersError && (
                <p className="text-xs text-amber-400">{ordinalUsersError}</p>
              )}
              {!loadingOrdinalUsers && !ordinalUsersError && ordinalUsers.length === 0 && (
                <p className="text-xs text-stone-500">No users returned.</p>
              )}
              <ul className="max-h-40 space-y-1 overflow-y-auto rounded border border-stone-800 p-2">
                {ordinalUsers.map((u: any) => {
                  const uid = u.id || u.userId;
                  if (!uid) return null;
                  const name = [u.firstName, u.lastName].filter(Boolean).join(" ") || "User";
                  return (
                    <li key={uid}>
                      <label className="flex cursor-pointer items-center gap-2 text-xs text-stone-300">
                        <input
                          type="checkbox"
                          checked={pushApproverIds.has(uid)}
                          onChange={() => toggleApprover(uid)}
                        />
                        <span>
                          {name} {u.email ? <span className="text-stone-500">({u.email})</span> : null}
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              {pushApproverIds.size > 0 && (
                <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-stone-400">
                  <input
                    type="checkbox"
                    checked={pushApprovalsBlocking}
                    onChange={(e) => setPushApprovalsBlocking(e.target.checked)}
                  />
                  Blocking approvals
                </label>
              )}
            </div>
            <div className="flex justify-end gap-2 border-t border-stone-800 pt-3">
              <button
                type="button"
                onClick={() => setPushAllOpen(false)}
                className="rounded bg-stone-800 px-3 py-1.5 text-xs text-stone-300 hover:bg-stone-700"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleConfirmPushAll()}
                disabled={actionInProgress === "__push_all__"}
                className="rounded bg-cyan-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-cyan-700 disabled:opacity-50"
              >
                {actionInProgress === "__push_all__" ? "Pushing…" : `Push all (${posts.length})`}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-stone-300">
          {posts.length} post{posts.length !== 1 ? "s" : ""}
        </h3>
        <div className="flex flex-wrap items-center gap-2">
          {/* "Push all to Ordinal" removed 2026-04-23 — Virio churning
              off Ordinal; backend returns 410 Gone. Drafts still land
              in Amphoreus local_posts and the calendar; publishing will
              move to the replacement pipeline. */}
          <button onClick={onRefresh} className="text-xs text-stone-500 hover:text-stone-300">
            Refresh
          </button>
        </div>
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
              {/* Fact-checked pill — shown when Castorice edited the draft */}
              {post.pre_revision_content && (
                <div className="mb-2 flex items-center gap-2">
                  <span className="rounded bg-stone-800 px-2 py-0.5 text-[10px] text-stone-400">
                    Fact-checked
                  </span>
                </div>
              )}

              {/* Pre-revision toggle + revert-to-original */}
              {post.pre_revision_content && (
                <details className="mb-3">
                  <summary className="cursor-pointer text-xs text-stone-500 hover:text-stone-300">
                    Show original draft (before fact-checking)
                  </summary>
                  <div className="mt-2 max-h-[min(50vh,20rem)] overflow-y-auto rounded border border-stone-700/50 bg-stone-950/30 p-3">
                    <pre className="whitespace-pre-wrap break-words text-sm text-stone-400">
                      {post.pre_revision_content}
                    </pre>
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void handleRevertToOriginal(post.id)}
                      disabled={actionInProgress === post.id}
                      className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700 disabled:opacity-40"
                      title="Replace the current content with Stelle's original. The current version stays in the revision history."
                    >
                      ↺ Revert to Stelle&rsquo;s original
                    </button>
                    <span className="text-[11px] text-stone-500">
                      Keeps the current version in revision history.
                    </span>
                  </div>
                </details>
              )}

              {/* Body + inline-comment rail. Two columns when inline
                  comments exist; full-width otherwise. The body wraps
                  each inline-comment selection in a <mark> element so
                  the operator can SEE which passages have feedback,
                  and click-syncs to the rail card. onMouseUp captures
                  fresh selections to seed the next inline comment. */}
              {(() => {
                const allComments = commentsByPost[post.id] ?? [];
                const inlineComments = allComments
                  .filter((c) => c.selection_start != null && c.selection_end != null && !!c.selected_text)
                  .sort((a, b) => (a.selection_start! - b.selection_start!));
                const postWide = allComments.filter((c) => !inlineComments.includes(c));
                const content = post.content || "";

                // Build the body as a list of plain-text spans + <mark>
                // wrappers. Validity check: the text at the stored
                // offsets must still equal the stored selected_text —
                // operator edits can desync them, in which case we
                // skip the highlight (the comment still lives in the
                // rail, badged 'stale').
                type Piece = { text: string; commentId?: string; stale?: boolean };
                const pieces: Piece[] = [];
                {
                  let cursor = 0;
                  for (const c of inlineComments) {
                    const s = c.selection_start ?? -1;
                    const e = c.selection_end ?? -1;
                    if (s < cursor || e <= s || e > content.length) continue; // overlap / past edits
                    if (content.slice(s, e) !== c.selected_text) continue;     // stale; skip mark
                    if (s > cursor) pieces.push({ text: content.slice(cursor, s) });
                    pieces.push({ text: content.slice(s, e), commentId: c.id });
                    cursor = e;
                  }
                  if (cursor < content.length) pieces.push({ text: content.slice(cursor) });
                }
                const isStale = (c: Comment) =>
                  c.selection_start == null || c.selection_end == null ||
                  content.slice(c.selection_start, c.selection_end) !== c.selected_text;

                return (
                  <div className="mb-3 flex gap-3">
                    {/* Left: body with marks */}
                    <div className="flex-1 max-h-[min(70vh,32rem)] overflow-y-auto rounded border border-stone-800/80 bg-stone-950/50 p-3">
                      <pre
                        className="whitespace-pre-wrap break-words text-sm text-stone-200"
                        onContextMenu={(e) => openInlineCommentPopup(e, post.id, content)}
                      >
                        {pieces.length > 0
                          ? pieces.map((p, i) =>
                              p.commentId ? (
                                <mark
                                  key={`mark-${i}`}
                                  id={`cmt-mark-${post.id}-${p.commentId}`}
                                  onClick={(e) => {
                                    // Left-click → edit/delete popup
                                    // anchored on the mark itself. The
                                    // comment's body text is looked up
                                    // on the fly from commentsByPost so
                                    // an edit made in the rail reflects
                                    // here without extra state sync.
                                    const body =
                                      (commentsByPost[post.id] ?? []).find(
                                        (c) => c.id === p.commentId,
                                      )?.body ?? "";
                                    openMarkPopup(e, post.id, p.commentId!, body);
                                  }}
                                  onMouseEnter={() => setHighlightedCommentId(p.commentId!)}
                                  onMouseLeave={() => setHighlightedCommentId(null)}
                                  className={`cursor-pointer rounded px-0.5 transition-colors ${
                                    highlightedCommentId === p.commentId
                                      ? "bg-amber-300/70 text-stone-950"
                                      : "bg-amber-300/25 text-stone-100 hover:bg-amber-300/45"
                                  }`}
                                  title="Click to edit or delete this comment"
                                >
                                  {p.text}
                                </mark>
                              ) : (
                                <span key={`txt-${i}`}>{p.text}</span>
                              )
                            )
                          : content}
                      </pre>
                    </div>

                    {/* Right: inline-comment rail. Only renders when
                        inline comments exist — no point in a blank
                        column otherwise. */}
                    {inlineComments.length > 0 && (
                      <aside className="w-72 shrink-0 max-h-[min(70vh,32rem)] overflow-y-auto rounded border border-stone-800/80 bg-stone-950/30 p-2">
                        <div className="mb-1.5 flex items-center justify-between px-1">
                          <span className="text-[10px] font-semibold uppercase tracking-wide text-stone-500">
                            Inline comments
                          </span>
                          <span className="rounded-full bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-200">
                            {inlineComments.filter((c) => !c.resolved).length}/{inlineComments.length}
                          </span>
                        </div>
                        <div className="space-y-1.5">
                          {inlineComments.map((c) => {
                            const stale = isStale(c);
                            const editing = editingCommentId === c.id;
                            return (
                              <div
                                key={c.id}
                                id={`cmt-card-${post.id}-${c.id}`}
                                onMouseEnter={() => setHighlightedCommentId(c.id)}
                                onMouseLeave={() => setHighlightedCommentId(null)}
                                onClick={() => !stale && scrollToCommentMark(post.id, c.id)}
                                className={`cursor-pointer rounded border p-1.5 text-[11px] transition-colors ${
                                  c.resolved
                                    ? "border-stone-800 bg-stone-900/30 text-stone-500"
                                    : highlightedCommentId === c.id
                                      ? "border-amber-500/60 bg-amber-500/10 text-stone-100"
                                      : "border-stone-700 bg-stone-900/70 text-stone-200"
                                }`}
                              >
                                <div
                                  className={`mb-1 truncate text-[10px] italic ${stale ? "text-stone-600 line-through" : "text-emerald-400/80"}`}
                                  title={c.selected_text || ""}
                                >
                                  &ldquo;{c.selected_text}&rdquo;
                                  {stale && <span className="ml-1 not-italic">stale</span>}
                                </div>
                                {editing ? (
                                  <>
                                    <textarea
                                      value={editingCommentText}
                                      onChange={(e) => setEditingCommentText(e.target.value)}
                                      rows={3}
                                      onClick={(e) => e.stopPropagation()}
                                      className="w-full rounded border border-stone-600 bg-stone-950 px-1.5 py-1 text-[11px] text-stone-100 focus:border-stone-400 focus:outline-none"
                                    />
                                    <div className="mt-1 flex justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                                      <button
                                        type="button"
                                        onClick={() => { setEditingCommentId(null); setEditingCommentText(""); }}
                                        className="text-[10px] text-stone-500 hover:text-stone-300"
                                      >
                                        cancel
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() => void handleSaveEditComment(post.id, c.id)}
                                        disabled={!editingCommentText.trim()}
                                        className="rounded bg-stone-700 px-2 py-0.5 text-[10px] font-medium text-stone-100 hover:bg-stone-600 disabled:opacity-40"
                                      >
                                        save
                                      </button>
                                    </div>
                                  </>
                                ) : (
                                  <div className="whitespace-pre-wrap break-words">{c.body}</div>
                                )}
                                <div className="mt-1 flex items-center justify-between text-[9px] text-stone-500" onClick={(e) => e.stopPropagation()}>
                                  <span className="truncate">
                                    {c.author_name || c.author_email || "operator"}
                                  </span>
                                  {!editing && (
                                    <span className="flex shrink-0 gap-1.5">
                                      <button
                                        type="button"
                                        onClick={() => startEditingComment(c)}
                                        className="text-stone-500 hover:text-stone-300"
                                        title="Edit"
                                      >
                                        edit
                                      </button>
                                      {!c.resolved && (
                                        <button
                                          type="button"
                                          onClick={() => void handleResolveComment(post.id, c.id)}
                                          className="text-stone-500 hover:text-emerald-400"
                                        >
                                          resolve
                                        </button>
                                      )}
                                      <button
                                        type="button"
                                        onClick={() => void handleDeleteComment(post.id, c.id)}
                                        className="text-stone-500 hover:text-red-400"
                                      >
                                        delete
                                      </button>
                                    </span>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </aside>
                    )}
                  </div>
                );
              })()}

              {/* Three-way split (2026-04-29):
                    1. why_post  — Castorice's strategic-fit verdict
                       (~50 words). Visible by default — glanceable
                       rationale the operator reads before push.
                    2. process_notes — Stelle's audit trail. Hidden
                       behind a "Show process notes" expander; debugging
                       only, not review-time judgment.
                    3. fact_check_report + citation_comments — Castorice
                       fact-check transcript. Own collapsed expander.

                    Pre-split rows have the entire mishmash in why_post
                    and NULL on the other two columns; we just render
                    that as-is (operator sees the long version on legacy
                    drafts, the new short version on fresh ones). */}
              {(() => {
                const whyPost = (post.why_post || "").trim();
                const processNotes = (post.process_notes || "").trim();
                const factCheckReport = (post.fact_check_report || "").trim();

                let citations: string[] = [];
                const cc = post.citation_comments;
                if (cc) {
                  if (Array.isArray(cc)) {
                    citations = cc.filter((x: unknown): x is string => typeof x === "string");
                  } else if (typeof cc === "string") {
                    // Stored as JSON string in local_posts.citation_comments.
                    try {
                      const parsed = JSON.parse(cc);
                      if (Array.isArray(parsed)) {
                        citations = parsed.filter((x: unknown): x is string => typeof x === "string");
                      }
                    } catch {
                      /* not valid JSON — treat as a single string entry */
                      if (cc.trim()) citations = [cc.trim()];
                    }
                  }
                }

                if (!whyPost && !processNotes && !factCheckReport && citations.length === 0) {
                  return null;
                }

                return (
                  <div className="mb-3 space-y-2">
                    {/* Operator-facing rationale — visible by default */}
                    {whyPost && (
                      <div className="rounded border border-stone-800 bg-stone-950/40 px-3 py-2 text-xs text-stone-300">
                        <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-stone-500">
                          Why we&apos;re posting this
                        </div>
                        <div className="whitespace-pre-wrap">{whyPost}</div>
                      </div>
                    )}

                    {/* Audit trail — collapsed by default */}
                    {processNotes && (
                      <details className="rounded border border-stone-800 bg-stone-950/40">
                        <summary className="cursor-pointer px-3 py-1.5 text-[11px] text-stone-500 hover:text-stone-300 list-none">
                          Show process notes
                        </summary>
                        <div className="border-t border-stone-800/70 px-3 py-2 text-xs text-stone-400 whitespace-pre-wrap">
                          {processNotes}
                        </div>
                      </details>
                    )}

                    {/* Fact-check + citations — own collapsed expander */}
                    {(factCheckReport || citations.length > 0) && (
                      <details className="rounded border border-stone-800 bg-stone-950/40">
                        <summary className="cursor-pointer px-3 py-1.5 text-xs text-stone-400 hover:text-stone-200 list-none">
                          <span className="font-medium">Castorice fact-check</span>
                          {citations.length > 0 && (
                            <span className="ml-2 text-[10px] text-stone-500">
                              {citations.length} citation{citations.length !== 1 ? "s" : ""}
                            </span>
                          )}
                        </summary>
                        <div className="space-y-3 border-t border-stone-800/70 px-3 py-2 text-xs text-stone-300">
                          {factCheckReport && (
                            <div className="whitespace-pre-wrap">{factCheckReport}</div>
                          )}
                          {citations.length > 0 && (
                            <div>
                              <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-stone-500">
                                Citations ({citations.length})
                              </div>
                              <ul className="space-y-2">
                                {citations.map((c, i) => (
                                  <li
                                    key={i}
                                    className="rounded border border-stone-800 bg-stone-900/60 p-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap"
                                  >
                                    {c}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </details>
                    )}
                  </div>
                );
              })()}

              {/* Post-wide comments panel (collapsible). Inline ones
                  already live in the right rail above; this section is
                  for free-form notes that don't anchor to any passage. */}
              {(() => {
                const allComments = commentsByPost[post.id] ?? [];
                const postWideComments = allComments.filter(
                  (c) => c.selection_start == null || c.selection_end == null,
                );
                const inlineCount = allComments.length - postWideComments.length;
                const openWide = postWideComments.filter((c) => !c.resolved).length;
                const isOpen = openCommentsFor === post.id;
                return (
                  <div className="mb-3">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => toggleComments(post.id)}
                        className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700"
                      >
                        {isOpen ? "Hide" : "Show"} post-wide comments
                        {postWideComments.length > 0 && (
                          <span className={`ml-1.5 rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${openWide > 0 ? "bg-amber-500/30 text-amber-200" : "bg-stone-700 text-stone-400"}`}>
                            {openWide > 0 ? `${openWide} open` : postWideComments.length}
                          </span>
                        )}
                      </button>
                      {inlineCount > 0 && (
                        <span className="text-[10px] text-stone-500">
                          + {inlineCount} inline {inlineCount === 1 ? "comment" : "comments"} in side rail
                        </span>
                      )}
                      {/* (2026-04-23) The "Commenting on selection…"
                          blurb used to live here. Removed once the
                          right-click inline-comment popup shipped —
                          the blurb was a leftover artifact of the
                          auto-staging onMouseUp path that no longer
                          exists. Inline comments now route exclusively
                          through the popup; the composer below is
                          post-wide only. */}
                    </div>
                    {isOpen && (
                      <div className="mt-2 space-y-2 rounded border border-stone-800 bg-stone-950/40 p-3">
                        {postWideComments.length === 0 ? (
                          <p className="text-[11px] italic text-stone-500">
                            No post-wide comments yet. Add one below — or highlight text in the post above to leave an inline comment.
                          </p>
                        ) : (
                          postWideComments.map((c) => {
                            const editing = editingCommentId === c.id;
                            return (
                              <div
                                key={c.id}
                                className={`rounded border p-2 text-xs ${c.resolved ? "border-stone-800 bg-stone-900/30 text-stone-500" : "border-stone-700 bg-stone-900/70 text-stone-200"}`}
                              >
                                {editing ? (
                                  <>
                                    <textarea
                                      value={editingCommentText}
                                      onChange={(e) => setEditingCommentText(e.target.value)}
                                      rows={4}
                                      className="w-full rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-xs text-stone-100 focus:border-stone-400 focus:outline-none"
                                    />
                                    <div className="mt-1 flex justify-end gap-2">
                                      <button
                                        type="button"
                                        onClick={() => { setEditingCommentId(null); setEditingCommentText(""); }}
                                        className="text-[11px] text-stone-500 hover:text-stone-300"
                                      >
                                        cancel
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() => void handleSaveEditComment(post.id, c.id)}
                                        disabled={!editingCommentText.trim()}
                                        className="rounded bg-stone-700 px-2.5 py-0.5 text-[11px] font-medium text-stone-100 hover:bg-stone-600 disabled:opacity-40"
                                      >
                                        save
                                      </button>
                                    </div>
                                  </>
                                ) : (
                                  <div className="whitespace-pre-wrap">{c.body}</div>
                                )}
                                <div className="mt-1 flex items-center justify-between text-[10px] text-stone-500">
                                  <span>
                                    {c.author_name || c.author_email || "operator"}
                                    {" · "}
                                    {new Date(c.created_at).toLocaleString()}
                                    {c.resolved && <span className="ml-2 text-stone-600">resolved</span>}
                                  </span>
                                  {!editing && (
                                    <span className="flex gap-2">
                                      <button
                                        type="button"
                                        onClick={() => startEditingComment(c)}
                                        className="text-stone-500 hover:text-stone-300"
                                      >
                                        edit
                                      </button>
                                      {!c.resolved && (
                                        <button
                                          type="button"
                                          onClick={() => void handleResolveComment(post.id, c.id)}
                                          className="text-stone-500 hover:text-emerald-400"
                                        >
                                          resolve
                                        </button>
                                      )}
                                      <button
                                        type="button"
                                        onClick={() => void handleDeleteComment(post.id, c.id)}
                                        className="text-stone-500 hover:text-red-400"
                                      >
                                        delete
                                      </button>
                                    </span>
                                  )}
                                </div>
                              </div>
                            );
                          })
                        )}

                        <div className="pt-1">
                          <textarea
                            value={composerText[post.id] || ""}
                            onChange={(e) =>
                              setComposerText((m) => ({ ...m, [post.id]: e.target.value }))
                            }
                            rows={2}
                            placeholder="Leave a post-wide comment. Cyrene will see unresolved comments on the next rewrite."
                            className="w-full rounded border border-stone-700 bg-stone-950 px-2 py-1.5 text-xs text-stone-100 focus:border-stone-500 focus:outline-none"
                          />
                          <div className="mt-1 flex items-center justify-end gap-2">
                            <button
                              type="button"
                              onClick={() => void handleAddComment(post.id)}
                              disabled={
                                actionInProgress === post.id ||
                                !(composerText[post.id] || "").trim()
                              }
                              className="rounded bg-stone-700 px-3 py-1 text-xs font-medium text-stone-100 hover:bg-stone-600 disabled:opacity-40"
                            >
                              Add comment
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
              {(pendingImageByPost[post.id] || post.linked_image_id) && (
                <div className="mb-3 rounded-lg border border-amber-900/50 bg-stone-950/50 p-3">
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <p className="text-xs font-medium text-amber-200/90">LinkedIn / Ordinal image</p>
                    {post.linked_image_id && (
                      <span
                        className="rounded bg-amber-950/80 px-2 py-0.5 text-[10px] font-medium text-amber-400"
                        title="This draft’s PNG will be uploaded to Ordinal on push when PUBLIC_BASE_URL is set."
                      >
                        Approved for push
                      </span>
                    )}
                    {pendingImageByPost[post.id] &&
                      pendingImageByPost[post.id] !== post.linked_image_id && (
                        <span className="rounded bg-stone-800 px-2 py-0.5 text-[10px] text-stone-400">
                          New preview — click &quot;Use for Ordinal&quot; to save
                        </span>
                      )}
                  </div>
                  <img
                    src={imagesApi.getUrl(
                      company,
                      pendingImageByPost[post.id] || post.linked_image_id
                    )}
                    alt=""
                    className="mb-3 max-h-64 max-w-full rounded border border-stone-700"
                  />
                  <label className="block text-xs text-stone-400">
                    Revision notes for Phainon (concrete visual changes work best)
                    <textarea
                      value={imageFeedbackByPost[post.id] ?? ""}
                      onChange={(e) =>
                        setImageFeedbackByPost((p) => ({ ...p, [post.id]: e.target.value }))
                      }
                      rows={3}
                      className="mt-1 w-full rounded border border-stone-600 bg-stone-950 px-2 py-1.5 text-sm text-stone-200"
                      placeholder="e.g. warmer palette, less text on the image, tighter crop…"
                    />
                  </label>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => void handleRegenerateWithFeedback(post)}
                      disabled={generatingImageFor === post.id || actionInProgress === post.id}
                      className="rounded bg-stone-800 px-2.5 py-1 text-xs text-amber-300 hover:bg-stone-700 disabled:opacity-50"
                    >
                      {generatingImageFor === post.id ? "Regenerating…" : "Regenerate with notes"}
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleApproveLinkedImage(post)}
                      disabled={actionInProgress === post.id}
                      className="rounded bg-amber-900/60 px-2.5 py-1 text-xs font-medium text-amber-100 hover:bg-amber-900 disabled:opacity-50"
                    >
                      Use for Ordinal
                    </button>
                  </div>
                  <p className="mt-2 text-[11px] text-stone-500">
                    Ordinal fetches the image from a public URL — set{" "}
                    <code className="text-stone-400">PUBLIC_BASE_URL</code> on the API host (tunnel or prod) so
                    uploads succeed.
                  </p>
                </div>
              )}
              <div className="flex flex-wrap items-center gap-2 border-t border-stone-800 pt-3">
                {post.ordinal_post_id && (
                  <span
                    className="max-w-full truncate rounded bg-stone-800/80 px-2 py-0.5 font-mono text-[10px] text-stone-400"
                    title={`Latest Ordinal post id (updates if you re-push): ${post.ordinal_post_id}`}
                  >
                    Ordinal: {post.ordinal_post_id}
                  </span>
                )}
                {!post.ordinal_post_id && post.match_kind === "auto" && post.match_provider_urn && (
                  <span
                    className="max-w-full truncate rounded bg-emerald-900/40 px-2 py-0.5 font-mono text-[10px] text-emerald-300"
                    title={`Attributed via semantic match (cosine=${typeof post.match_similarity === "number" ? post.match_similarity.toFixed(3) : "?"}) — draft was likely copy-pasted into Lineage and posted without Ordinal.`}
                  >
                    Matched: {post.match_provider_urn}
                  </span>
                )}
                {!post.ordinal_post_id && post.match_kind === "ambiguous" && (
                  <span
                    className="rounded bg-amber-900/40 px-2 py-0.5 text-[10px] text-amber-300"
                    title={`Semantic match was borderline (top sim=${typeof post.match_similarity === "number" ? post.match_similarity.toFixed(3) : "?"}). Operator disambiguation needed.`}
                  >
                    Match: ambiguous
                  </span>
                )}
                {citationData[post.id] && (
                  <span
                    title={`${citationData[post.id].length} source annotation${citationData[post.id].length !== 1 ? "s" : ""} ready — will be posted as Ordinal comments on push`}
                    className="rounded bg-emerald-900/40 px-2 py-0.5 text-xs font-medium text-emerald-400"
                  >
                    Annotated
                  </span>
                )}
                <button
                  onClick={() => { setEditingId(post.id); setEditText(post.content || ""); }}
                  className="rounded bg-stone-800 px-2.5 py-1 text-xs text-stone-300 hover:bg-stone-700"
                >
                  Edit
                </button>
                <button
                  onClick={() => void handleCopyPost(post.id, post.content || "")}
                  disabled={!post.content}
                  title="Copy the post body to the clipboard."
                  className={`rounded px-2.5 py-1 text-xs ${
                    copiedPostId === post.id
                      ? "bg-emerald-900/40 text-emerald-300"
                      : "bg-stone-800 text-stone-300 hover:bg-stone-700"
                  } disabled:opacity-40`}
                >
                  {copiedPostId === post.id ? "Copied!" : "Copy"}
                </button>
                {/* Rewrite — pulls every UNRESOLVED inline + post-wide
                    comment on this post and feeds them to the rewrite
                    agent (Cyrene.rewrite_single_post). The backend
                    persists the new content as a revision with source
                    "rewrite_with_feedback" and returns the rewrite to
                    the UI, which opens the edit panel pre-filled so
                    the operator can refine further before final save. */}
                <button
                  onClick={() => handleRewrite(post)}
                  disabled={actionInProgress === post.id}
                  title="Rewrite this draft using every unresolved inline + post-wide comment as guidance. Saves the rewrite as a new revision; the edit panel opens with the new text so you can refine before approving."
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
                {/* Generate Image — deprecated 2026-05-01. The image-
                    generation pipeline runs but the resulting images
                    aren't being used downstream (Ordinal churn closed
                    the publishing path that consumed them). The
                    handler + backend stay in place so re-enabling is
                    a one-line `{false &&` flip if/when image attach
                    on the next publishing surface lands. */}
                {false && (
                  <button
                    onClick={() => handleGenerateImage(post)}
                    disabled={generatingImageFor === post.id || actionInProgress === post.id}
                    className="rounded bg-stone-800 px-2.5 py-1 text-xs text-amber-400 hover:bg-stone-700 disabled:opacity-50"
                  >
                    {generatingImageFor === post.id ? "Generating..." : "Generate Image"}
                  </button>
                )}
                {/* Per-post "Push to Ordinal" + "Push Original" buttons
                    removed 2026-04-23 — Virio churning off Ordinal.
                    Backend returns 410 Gone on any remaining caller.
                    Drafts stay in local_posts; when the replacement
                    publishing pipeline ships, the button will land
                    here with a new label + destination. */}
                <div className="flex-1" />
                <span className="text-xs text-stone-600">
                  {post.status || "draft"}
                  {post.created_at ? ` \u00b7 ${new Date(post.created_at * 1000).toLocaleDateString()}` : ""}
                </span>
                {/* Pair state chip — set when the semantic match-back
                    worker linked this draft to a published LinkedIn
                    post. Auto-pairs run after every Apify scrape;
                    operators don't trigger pairing manually anymore.
                    Tooltip carries provenance metadata. */}
                {post.matched_provider_urn && (
                  <span
                    title={[
                      `matched_provider_urn: ${post.matched_provider_urn}`,
                      `match_method: ${post.match_method || "?"}`,
                      post.match_similarity != null
                        ? `match_similarity: ${post.match_similarity}`
                        : null,
                      post.matched_at ? `matched_at: ${post.matched_at}` : null,
                    ]
                      .filter(Boolean)
                      .join("\n")}
                    className="rounded bg-emerald-950/40 px-2 py-1 text-xs text-emerald-300"
                  >
                    Paired
                    {post.match_method ? ` \u00b7 ${post.match_method}` : ""}
                  </span>
                )}
                <button
                  onClick={() => void handleReject(post.id)}
                  disabled={actionInProgress === post.id || post.status === "rejected"}
                  title="Mark as rejected by the client. Preserves the draft + comments as a learning signal."
                  className="rounded px-2 py-1 text-xs text-amber-500 hover:bg-amber-950 disabled:opacity-50"
                >
                  {post.status === "rejected" ? "Rejected" : "Reject"}
                </button>
                <button
                  onClick={() => handleDelete(post.id)}
                  disabled={actionInProgress === post.id}
                  title="Permanently delete the draft. Removes the row and all paired comments from Amphoreus. Use 'Reject' instead if the client turned the draft down — Reject keeps it as a learning signal."
                  className="rounded px-2 py-1 text-xs text-red-500 hover:bg-red-950 disabled:opacity-50"
                >
                  Delete
                </button>
              </div>

              {/* Published-version expander — visible when the
                  semantic match-back worker linked this draft to a
                  published post AND list_posts joined the body in.
                  Operators click to inspect the draft → published
                  diff (what the client actually shipped). */}
              {post.matched_provider_urn && post.published_post_text && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-emerald-400 hover:text-emerald-300">
                    Show published version
                    {post.published_posted_at
                      ? ` \u00b7 ${new Date(post.published_posted_at).toLocaleDateString()}`
                      : ""}
                    {post.published_reactions != null
                      ? ` \u00b7 ${post.published_reactions} reactions`
                      : ""}
                  </summary>
                  <div className="mt-2 whitespace-pre-wrap rounded border border-emerald-900/40 bg-emerald-950/20 p-3 text-sm text-stone-200">
                    {post.published_post_text}
                  </div>
                </details>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}



