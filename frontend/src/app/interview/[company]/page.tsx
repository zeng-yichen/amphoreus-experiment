"use client";

/**
 * Interview Prep — merged Interview Briefing + Interview Companion.
 *
 * Two phases on one page:
 *
 *   Phase A (Prep):
 *     - Click "Generate briefing" → runs Aglaea via /api/briefings/generate,
 *       streams the progress to a log panel, then populates the briefing panel
 *       with the resulting markdown.
 *     - Existing briefing (if any) is loaded on mount and shown immediately.
 *
 *   Phase B (Live):
 *     - Once a briefing exists and BlackHole is detected, "Start recording"
 *       kicks off Tribbie via /api/interview/start. Live transcript streams
 *       into the left panel, follow-up suggestions appear on the right.
 *
 * The old standalone /briefings/{company} route is gone; this page owns the
 * full interview-prep lifecycle.
 */

import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { briefingsApi, cyreneApi, ghostwriterApi, interviewApi, transcriptsApi, type MeetingSubtype } from "@/lib/api";
import { startInterviewCapture, type CaptureHandle } from "@/lib/interview-capture";

interface TranscriptSegment {
  text: string;
  timestamp: number;
}

interface Suggestion {
  text: string;
  timestamp: number;
}

interface BriefingLogLine {
  type: string;
  text: string;
  timestamp: number;
}

export default function InterviewPrepPage() {
  const params = useParams();
  const company = params.company as string;


  // --- Live interview (Tribbie) state ---
  const [isRecording, setIsRecording] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("Ready");
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [trashed, setTrashed] = useState(false);

  // Post-session "upload to Amphoreus" modal state. Opens automatically
  // when the Tribbie session completes and the transcript survives
  // (not trashed). User picks meeting_subtype + confirms, we POST the
  // stitched transcript to the mirror paste endpoint. The write lands
  // in Amphoreus Supabase so cyrene.fly.dev + Stelle see it — even
  // though the recording happened on the operator's laptop.
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadSubtype, setUploadSubtype] = useState<MeetingSubtype>("content_interview");
  const [uploadDescription, setUploadDescription] = useState("");
  const [uploadingToMirror, setUploadingToMirror] = useState(false);
  const [uploadedToMirror, setUploadedToMirror] = useState<{ id: string; filename: string } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Browser-capture support detection. Tribbie now captures audio in
  // the browser via getDisplayMedia (tab/system audio) + getUserMedia
  // (mic) — no BlackHole, no local install. Safari support for
  // getDisplayMedia audio is still partial, so we feature-detect and
  // show a short browser-compat warning when unsupported. Runs once
  // on mount; no backend call needed.
  const [canCaptureInBrowser, setCanCaptureInBrowser] = useState<boolean | null>(null);
  useEffect(() => {
    if (typeof navigator === "undefined") return;
    const ok =
      !!navigator.mediaDevices
      && typeof navigator.mediaDevices.getDisplayMedia === "function"
      && typeof navigator.mediaDevices.getUserMedia === "function";
    setCanCaptureInBrowser(ok);
  }, []);

  // --- Briefing (Aglaea) state ---
  const [hasBriefing, setHasBriefing] = useState<boolean | null>(null);
  const [briefingContent, setBriefingContent] = useState<string | null>(null);
  const [isGeneratingBrief, setIsGeneratingBrief] = useState(false);
  const [briefingLog, setBriefingLog] = useState<BriefingLogLine[]>([]);
  const [showBriefingLog, setShowBriefingLog] = useState(false);

  // --- Cyrene (Strategic Review) state ---
  const [cyreneBrief, setCyreneBrief] = useState<any>(null);
  const [isRunningCyrene, setIsRunningCyrene] = useState(false);
  const [cyreneLog, setCyreneLog] = useState<BriefingLogLine[]>([]);
  const [showCyreneLog, setShowCyreneLog] = useState(false);
  const [showCyreneBrief, setShowCyreneBrief] = useState(false);

  const transcriptRef = useRef<HTMLDivElement>(null);
  const briefingLogRef = useRef<HTMLDivElement>(null);
  // Browser-capture handle for the current session. Created in
  // handleStart, torn down in handleStop / on done / on error. Keeping
  // it in a ref (not state) so capture teardown doesn't race React
  // re-renders and we can always call stop() synchronously.
  const captureRef = useRef<CaptureHandle | null>(null);

  // Whether to capture the interviewer's mic alongside the tab audio.
  // Default: true (complete two-sided transcript, needs headphones).
  // False: tab-audio only (matches legacy BlackHole scope — interviewee
  // voice only, zero bleed risk, no headphones required). Persisted per
  // browser in localStorage so each teammate's preference sticks.
  const [captureMic, setCaptureMic] = useState<boolean>(true);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem("tribbie.captureMic");
    if (stored === "false") setCaptureMic(false);
  }, []);
  function toggleCaptureMic(next: boolean) {
    setCaptureMic(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("tribbie.captureMic", String(next));
    }
  }

  // Check briefing existence + content on mount. Audio capture is a
  // browser capability now (see canCaptureInBrowser above) — no
  // backend round-trip needed to decide whether we can record.
  useEffect(() => {
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
  }, [company]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  // Auto-scroll briefing log
  useEffect(() => {
    if (briefingLogRef.current) {
      briefingLogRef.current.scrollTop = briefingLogRef.current.scrollHeight;
    }
  }, [briefingLog]);

  // Load existing Cyrene brief on mount
  useEffect(() => {
    cyreneApi
      .getBrief(company)
      .then((brief) => {
        setCyreneBrief(brief);
        setShowCyreneBrief(true);
      })
      .catch(() => setCyreneBrief(null));
  }, [company]);

  // Persist the live Cyrene job per-company so navigating away or closing
  // the tab doesn't lose the stream — on remount we rejoin.
  const activeCyreneKey = `amphoreus_cyrene_job_${company}`;

  const consumeCyreneStream = useCallback(
    async (jobId: string, afterId = 0) => {
      try {
        for await (const data of cyreneApi.streamJob(jobId, afterId)) {
          const text =
            (data.data as { message?: string } | undefined)?.message ||
            (data.data as { output?: string } | undefined)?.output ||
            JSON.stringify(data.data).slice(0, 200);
          setCyreneLog((prev) => [
            ...prev,
            { type: data.type, text, timestamp: Date.now() },
          ]);

          if (data.type === "done") {
            localStorage.removeItem(activeCyreneKey);
            try {
              const output = (data.data as { output?: string } | undefined)?.output;
              if (output) {
                setCyreneBrief(JSON.parse(output));
              } else {
                const fresh = await cyreneApi.getBrief(company);
                setCyreneBrief(fresh);
              }
            } catch {
              try {
                const fresh = await cyreneApi.getBrief(company);
                setCyreneBrief(fresh);
              } catch { /* ignore */ }
            }
            setShowCyreneBrief(true);
            setShowCyreneLog(false);
            setIsRunningCyrene(false);
            return;
          }
          if (data.type === "error") {
            localStorage.removeItem(activeCyreneKey);
            setIsRunningCyrene(false);
            return;
          }
        }
        setIsRunningCyrene(false);
      } catch (e) {
        setCyreneLog((prev) => [
          ...prev,
          { type: "error", text: String(e), timestamp: Date.now() },
        ]);
        setIsRunningCyrene(false);
      }
    },
    [activeCyreneKey, company],
  );

  // Download the brief as a self-contained HTML file. Inline styles
  // mirror the on-screen visual (emerald accent for positive sections,
  // red for exhausted, stone for body) so the downloaded file looks
  // like what the operator sees in the panel. Browser save-as-PDF
  // works on the resulting HTML if the operator wants a PDF artifact —
  // we don't ship a PDF library because that's overkill for what is
  // essentially a structured memo.
  function handleDownloadBrief() {
    if (!cyreneBrief) return;
    const esc = (s: unknown) =>
      String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    const dateStr = cyreneBrief._computed_at
      ? new Date(cyreneBrief._computed_at).toLocaleDateString()
      : new Date().toLocaleDateString();
    const isoDate = (cyreneBrief._computed_at || new Date().toISOString()).slice(0, 10);

    const diagnosis = cyreneBrief.current_strategy_diagnosis || null;
    const themes = cyreneBrief.strategic_themes || [];
    const probes = cyreneBrief.topics_to_probe || [];
    const exhausted = cyreneBrief.topics_exhausted || [];
    const dms = cyreneBrief.dm_targets || [];
    const nextTrigger = cyreneBrief.next_run_trigger || {};
    const prose = cyreneBrief.prose || "";

    const diagnosisHtml = diagnosis
      ? (() => {
          const w = diagnosis.what_is_working || [];
          const b = diagnosis.what_is_broken || [];
          const s = diagnosis.blind_spots || [];
          const block = (label: string, klass: "positive" | "negative" | "warning", items: any[], renderer: (x: any) => string) =>
            items.length === 0
              ? ""
              : `<div class="diag-block ${klass}"><h3 class="diag-label">${esc(label)} <span class="count">(${items.length})</span></h3><ul>${items.map(renderer).join("")}</ul></div>`;
          return [
            block(
              "What's working",
              "positive",
              w,
              (x) => `<li class="item"><div class="item-title positive">${esc(x.pattern)}</div>${x.evidence ? `<p class="item-body">${esc(x.evidence)}</p>` : ""}</li>`,
            ),
            block(
              "What's broken",
              "negative",
              b,
              (x) => `<li class="item"><div class="item-title negative">${esc(x.pattern)}</div>${x.evidence ? `<p class="item-body">${esc(x.evidence)}</p>` : ""}${x.consequence ? `<p class="item-italic">Consequence: ${esc(x.consequence)}</p>` : ""}</li>`,
            ),
            block(
              "Blind spots",
              "warning",
              s,
              (x) => `<li class="item"><div class="item-title warning">${esc(x.observation)}</div>${x.evidence ? `<p class="item-body">${esc(x.evidence)}</p>` : ""}</li>`,
            ),
          ].join("");
        })()
      : "";

    const themesHtml = themes
      .map(
        (t: any) => `
      <li class="item">
        <div class="item-title positive">${esc(t.theme)}${
          t.addresses ? ` <span class="addr-tag">↳ ${esc(t.addresses)}</span>` : ""
        }</div>
        ${t.evidence ? `<p class="item-body">${esc(t.evidence)}</p>` : ""}
        ${t.arc ? `<p class="item-italic">Arc: ${esc(t.arc)}</p>` : ""}
      </li>`,
      )
      .join("");

    const probesHtml = probes
      .map(
        (p: any, i: number) => `
      <li class="item">
        <div class="item-title positive"><span class="num">${i + 1}.</span> ${esc(p.thread)}</div>
        ${p.why ? `<p class="item-body">${esc(p.why)}</p>` : ""}
        ${p.suggested_entry_point ? `<p class="item-italic">Entry: ${esc(p.suggested_entry_point)}</p>` : ""}
      </li>`,
      )
      .join("");

    const exhaustedHtml = exhausted
      .map(
        (e: any) => `
      <li class="item">
        <div class="item-title negative">${esc(e.pattern)}</div>
        ${e.evidence ? `<p class="item-body">${esc(e.evidence)}</p>` : ""}
      </li>`,
      )
      .join("");

    const dmsHtml = dms
      .map(
        (d: any) => `
      <li class="item">
        <div class="item-title positive">${esc(d.name)}${d.company ? ` <span class="muted">@ ${esc(d.company)}</span>` : ""}</div>
        ${d.headline ? `<p class="item-italic">${esc(d.headline)}</p>` : ""}
        ${d.suggested_angle ? `<p class="item-body">${esc(d.suggested_angle)}</p>` : ""}
        ${
          d.icp_score !== undefined || d.posts_engaged !== undefined
            ? `<p class="muted small">${d.icp_score !== undefined ? `ICP ${esc(d.icp_score)}` : ""}${
                d.icp_score !== undefined && d.posts_engaged !== undefined ? " · " : ""
              }${d.posts_engaged !== undefined ? `${esc(d.posts_engaged)} engagements` : ""}</p>`
            : ""
        }
      </li>`,
      )
      .join("");

    const sections: string[] = [];
    if (diagnosisHtml)
      sections.push(`<section class="spine"><h2>Current strategy — diagnosis</h2><div class="diag-grid">${diagnosisHtml}</div></section>`);
    if (themes.length)
      sections.push(`<section><h2>Strategic Themes <span class="count">(${themes.length})</span></h2><ul>${themesHtml}</ul></section>`);
    if (probes.length)
      sections.push(`<section><h2>Topics to Probe <span class="count">(${probes.length})</span></h2><ul>${probesHtml}</ul></section>`);
    if (dms.length)
      sections.push(`<section><h2>DM Targets <span class="count">(${dms.length})</span></h2><ul>${dmsHtml}</ul></section>`);
    if (exhausted.length)
      sections.push(`<section class="exhausted"><h2>Topics Exhausted <span class="count">(${exhausted.length})</span></h2><ul>${exhaustedHtml}</ul></section>`);
    if (prose)
      sections.push(`<section><h2>Strategic Narrative</h2><div class="prose">${esc(prose).replace(/\n/g, "<br/>")}</div></section>`);

    const trigBits: string[] = [];
    if (nextTrigger.condition) trigBits.push(`<strong>Condition:</strong> ${esc(nextTrigger.condition)}`);
    if (nextTrigger.or_after_days !== undefined) trigBits.push(`<strong>Max days:</strong> ${esc(nextTrigger.or_after_days)}`);
    if (nextTrigger.rationale) trigBits.push(`<strong>Rationale:</strong> ${esc(nextTrigger.rationale)}`);
    if (trigBits.length)
      sections.push(`<section><h2>Next Run Trigger</h2><div class="prose">${trigBits.join("<br/>")}</div></section>`);

    const metaFooter: string[] = [];
    if (cyreneBrief._turns_used) metaFooter.push(`${esc(cyreneBrief._turns_used)} turns`);
    if (cyreneBrief._cost_usd) metaFooter.push(`$${esc(cyreneBrief._cost_usd)}`);

    const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cyrene Brief — ${esc(company)}</title>
<style>
  :root {
    --emerald-50: #ecfdf5;
    --emerald-200: #a7f3d0;
    --emerald-600: #059669;
    --emerald-800: #065f46;
    --emerald-900: #064e3b;
    --red-200: #fecaca;
    --red-800: #991b1b;
    --red-900: #7f1d1d;
    --amber-200: #fde68a;
    --amber-800: #92400e;
    --amber-900: #78350f;
    --stone-50: #fafaf9;
    --stone-200: #e7e5e4;
    --stone-400: #a8a29e;
    --stone-500: #78716c;
    --stone-600: #57534e;
    --stone-700: #44403c;
    --stone-900: #1c1917;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--stone-50);
    color: var(--stone-900);
    line-height: 1.55;
    font-size: 14px;
    padding: 32px 16px;
  }
  .wrap { max-width: 760px; margin: 0 auto; }
  header { border-bottom: 1px solid var(--emerald-200); padding-bottom: 16px; margin-bottom: 24px; }
  header h1 { margin: 0 0 4px; font-size: 22px; color: var(--emerald-900); font-weight: 600; }
  header .sub { color: var(--emerald-600); font-size: 13px; }
  section {
    background: white;
    border: 1px solid var(--emerald-200);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 16px;
  }
  section.exhausted { border-color: var(--red-200); }
  section h2 { margin: 0 0 10px; font-size: 14px; font-weight: 600; color: var(--emerald-800); }
  section.exhausted h2 { color: var(--red-800); }
  section h2 .count { font-weight: 400; color: var(--stone-500); }
  ul { list-style: none; margin: 0; padding: 0; }
  .item { padding: 8px 0; border-bottom: 1px solid var(--stone-200); }
  .item:last-child { border-bottom: none; }
  .item-title { font-weight: 500; margin-bottom: 2px; }
  .item-title.positive { color: var(--emerald-900); }
  .item-title.negative { color: var(--red-900); }
  .item-title.warning { color: var(--amber-900); }
  .addr-tag {
    display: inline-block;
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 4px;
    background: var(--emerald-50);
    color: var(--emerald-600);
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 10px;
    font-weight: 400;
  }
  section.spine { border-color: var(--emerald-600); border-width: 2px; }
  .diag-grid { display: flex; flex-direction: column; gap: 12px; }
  .diag-block { padding-top: 4px; }
  .diag-block + .diag-block { border-top: 1px solid var(--stone-200); padding-top: 12px; }
  .diag-label {
    margin: 0 0 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .diag-block.positive .diag-label { color: var(--emerald-600); }
  .diag-block.negative .diag-label { color: var(--red-800); }
  .diag-block.warning  .diag-label { color: var(--amber-800); }
  .item-title .num { color: var(--stone-400); margin-right: 4px; }
  .item-body { margin: 2px 0; color: var(--stone-600); font-size: 13px; }
  .item-italic { margin: 2px 0; font-style: italic; color: var(--stone-500); font-size: 13px; }
  .muted { color: var(--stone-500); font-weight: 400; }
  .small { font-size: 12px; }
  .prose { color: var(--stone-700); white-space: pre-wrap; line-height: 1.65; }
  footer { margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--stone-200); color: var(--stone-500); font-size: 12px; display: flex; gap: 16px; }
  @media print {
    body { background: white; padding: 0; }
    section { break-inside: avoid; box-shadow: none; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Cyrene Strategic Brief</h1>
      <div class="sub">${esc(company)} · Computed ${esc(dateStr)}</div>
    </header>
    ${sections.join("\n")}
    ${metaFooter.length ? `<footer>${metaFooter.map((m) => `<span>${m}</span>`).join("")}</footer>` : ""}
  </div>
</body>
</html>`;

    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cyrene-brief-${company}-${isoDate}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function handleRunCyrene() {
    if (isRunningCyrene) return;
    setIsRunningCyrene(true);
    setCyreneLog([]);
    setShowCyreneLog(true);
    setShowCyreneBrief(false);

    try {
      const { job_id } = await cyreneApi.run(company);
      localStorage.setItem(activeCyreneKey, job_id);
      setCyreneLog((prev) => [
        ...prev,
        { type: "status", text: `Cyrene review started: ${job_id}`, timestamp: Date.now() },
      ]);
      await consumeCyreneStream(job_id, 0);
    } catch (e) {
      setCyreneLog((prev) => [
        ...prev,
        { type: "error", text: String(e), timestamp: Date.now() },
      ]);
      setIsRunningCyrene(false);
    }
  }

  // On mount, if a Cyrene review is in-flight for this company, rejoin it.
  useEffect(() => {
    const savedJobId = localStorage.getItem(activeCyreneKey);
    if (!savedJobId) return;
    let cancelled = false;

    (async () => {
      try {
        // Events are stored in a shared run_events table, so reuse the
        // ghostwriter runs/events endpoint to hydrate history.
        const res = await ghostwriterApi.getRunEvents(savedJobId);
        if (cancelled) return;
        const status = res.run?.status;
        if (status === "completed" || status === "failed") {
          localStorage.removeItem(activeCyreneKey);
          return;
        }
        setIsRunningCyrene(true);
        setShowCyreneLog(true);
        setShowCyreneBrief(false);
        const hist: BriefingLogLine[] = res.events.map((ev: any) => {
          const parsed =
            typeof ev.data === "string"
              ? (() => {
                  try {
                    return JSON.parse(ev.data);
                  } catch {
                    return {};
                  }
                })()
              : ev.data || {};
          const text =
            parsed?.message ||
            parsed?.output ||
            JSON.stringify(parsed).slice(0, 200);
          return {
            type: ev.event_type,
            text,
            timestamp: ev.timestamp * 1000,
          };
        });
        setCyreneLog([
          { type: "status", text: `Rejoining Cyrene review ${savedJobId}…`, timestamp: Date.now() },
          ...hist,
        ]);
        const maxId = res.events.reduce(
          (m: number, e: any) => (e.id > m ? e.id : m),
          0,
        );
        await consumeCyreneStream(savedJobId, maxId);
      } catch {
        localStorage.removeItem(activeCyreneKey);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeCyreneKey, consumeCyreneStream]);

  async function handleGenerateBriefing() {
    if (isGeneratingBrief) return;
    setIsGeneratingBrief(true);
    setBriefingLog([]);
    setShowBriefingLog(true);

    try {
      const { job_id } = await briefingsApi.generate(company, company);
      setBriefingLog((prev) => [
        ...prev,
        { type: "status", text: `Job started: ${job_id}`, timestamp: Date.now() },
      ]);

      for await (const data of briefingsApi.streamJob(job_id)) {
        const text = extractBriefingText(data);
        setBriefingLog((prev) => [
          ...prev,
          {
            type: data.type,
            text,
            timestamp: ((data.timestamp as number | undefined) ?? Date.now() / 1000) * 1000,
          },
        ]);

        if (data.type === "done") {
          const output = (data.data as { output?: string } | undefined)?.output || "";
          if (output) {
            setBriefingContent(output);
            setHasBriefing(true);
          } else {
            // Try to fetch the persisted briefing if the done event didn't include it.
            try {
              const res = await briefingsApi.get(company);
              setBriefingContent(res.content);
              setHasBriefing(true);
            } catch {
              /* ignore */
            }
          }
          setIsGeneratingBrief(false);
          setShowBriefingLog(false);
          return;
        }
        if (data.type === "error") {
          setIsGeneratingBrief(false);
          return;
        }
      }
      setIsGeneratingBrief(false);
    } catch (e) {
      setBriefingLog((prev) => [
        ...prev,
        { type: "error", text: String(e), timestamp: Date.now() },
      ]);
      setIsGeneratingBrief(false);
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
    setStatus("Requesting audio permissions…");

    // Order matters: ask the backend to register the job + audio source
    // first (so the WS has something to attach to), THEN prompt the
    // browser for mic + tab audio, THEN open the WS from the hook.
    //
    // The browser-permission prompts (getUserMedia, getDisplayMedia)
    // MUST run inside this click handler — browsers block them when
    // triggered asynchronously after the user gesture boundary.
    let jid: string;
    try {
      const { job_id } = await interviewApi.start(company);
      jid = job_id;
      setJobId(job_id);
    } catch (e) {
      setError(`Couldn't start session: ${String(e)}`);
      setIsRecording(false);
      setStatus("Error");
      return;
    }

    try {
      setStatus("Pick the call tab in the picker and check ‘Share tab audio’…");
      const capture = await startInterviewCapture(jid, { captureMic });
      captureRef.current = capture;
      capture.onError((err) => {
        setError(`Audio capture failed: ${err.message}`);
      });
      setStatus("Audio connected. Starting transcription…");
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setIsRecording(false);
      setStatus("Error");
      // Best-effort: ask the backend to tear down the already-started
      // session so we don't leave a zombie job waiting for audio that
      // will never arrive.
      try { await interviewApi.stop(jid, company); } catch {/* ignore */}
      return;
    }

    try {
      for await (const data of interviewApi.streamJob(jid)) {
        const ts = ((data.timestamp as number | undefined) ?? Date.now() / 1000) * 1000;

        if (data.type === "status") {
          setStatus((data.data as { message?: string } | undefined)?.message || "");
        } else if (data.type === "text_delta") {
          const text = (data.data as { text?: string } | undefined)?.text || "";
          if (text) setTranscript((prev) => [...prev, { text, timestamp: ts }]);
        } else if (data.type === "tool_result") {
          const suggestion = (data.data as { result?: string } | undefined)?.result || "";
          if (suggestion) {
            setSuggestions((prev) => [{ text: suggestion, timestamp: ts }, ...prev]);
          }
        } else if (data.type === "error") {
          setError((data.data as { message?: string } | undefined)?.message || "Unknown error");
          setIsRecording(false);
          setStatus("Error");
          await captureRef.current?.stop();
          captureRef.current = null;
          return;
        } else if (data.type === "done") {
          const filePath = (data.data as { output?: string } | undefined)?.output || null;
          const msg = (data.data as { message?: string } | undefined)?.message || "Session complete";
          setSavedPath(filePath);
          setIsRecording(false);
          setStatus(msg);
          await captureRef.current?.stop();
          captureRef.current = null;
          // Auto-populate a sensible default description then prompt
          // the operator to upload. They can cancel the modal if this
          // was a throwaway recording.
          const now = new Date();
          const dateStr = now.toISOString().slice(0, 10);
          setUploadDescription(`Tribbie session ${dateStr}`);
          setShowUploadModal(true);
          return;
        }
      }
      setIsRecording(false);
      await captureRef.current?.stop();
      captureRef.current = null;
    } catch (e) {
      setError(String(e));
      setIsRecording(false);
      setStatus("Error");
      await captureRef.current?.stop();
      captureRef.current = null;
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
    // Don't stop the capture here — let the SSE 'done' handler do it
    // so any last-second audio still makes it through before the WS
    // closes. If the backend never sends 'done' the stream loop will
    // exit and its finally branch tears the capture down.
  }

  async function handleUploadToAmphoreus() {
    // Stitch the in-memory transcript back into text. The on-disk .txt
    // Tribbie saved locally has the same content, but using the UI
    // state means the operator gets exactly what they saw — no
    // post-processing drift.
    const text = transcript
      .map((s) => s.text)
      .join("\n")
      .trim();
    if (!text) {
      setUploadError("Transcript is empty — nothing to upload.");
      return;
    }
    if (!uploadDescription.trim()) {
      setUploadError("Add a description before uploading.");
      return;
    }
    setUploadingToMirror(true);
    setUploadError(null);
    try {
      const res = await transcriptsApi.pasteToMirror(
        company,
        text,
        uploadDescription.trim(),
        undefined,             // userId — resolved server-side from the slug
        "transcript",          // kind — always meetings-mount for Tribbie
        uploadSubtype,
      );
      setUploadedToMirror({ id: res.id, filename: res.filename });
      setShowUploadModal(false);
      setStatus(`Uploaded to Amphoreus as ${uploadSubtype === "content_interview" ? "interview" : uploadSubtype}`);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploadingToMirror(false);
    }
  }

  const canStart =
    canCaptureInBrowser !== false
    && hasBriefing === true
    && !isGeneratingBrief;

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 border-b border-stone-200 bg-white px-6 py-3">
        <Link href="/home" className="text-sm text-stone-400 hover:text-stone-600">
          &larr;
        </Link>
        <h1 className="text-lg font-semibold">Cyrene</h1>
        <span className="rounded bg-stone-100 px-2 py-0.5 text-xs text-stone-600">{company}</span>

        {isRecording && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-red-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
            Recording
          </span>
        )}
        {isGeneratingBrief && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-indigo-600">
            <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-500" />
            Generating briefing…
          </span>
        )}

        <div className="flex-1" />

        {/* Cyrene strategic review */}
        {!isRecording && (
          <>
            {cyreneBrief && !showCyreneBrief && !showCyreneLog && (
              <button
                onClick={() => setShowCyreneBrief(true)}
                className="rounded-lg border border-emerald-300 bg-white px-4 py-1.5 text-sm font-medium text-emerald-700 transition-colors hover:bg-emerald-50"
              >
                Show strategic brief
              </button>
            )}
            <button
              onClick={handleRunCyrene}
              disabled={isRunningCyrene}
              className="rounded-lg border border-emerald-300 bg-emerald-50 px-4 py-1.5 text-sm font-medium text-emerald-700 transition-colors hover:bg-emerald-100 disabled:cursor-wait disabled:opacity-50"
            >
              {isRunningCyrene
                ? "Running review…"
                : cyreneBrief
                ? "Re-run strategic review"
                : "Run strategic review"}
            </button>
          </>
        )}

        {/* Progress report — opens a page that streams generation progress
            and swaps in the rendered HTML on completion. */}
        {!isRecording && (
          <button
            onClick={() => window.open(`/report/${company}`, "_blank")}
            className="rounded-lg border border-stone-300 bg-white px-4 py-1.5 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50"
          >
            Progress report
          </button>
        )}

        {/* Record / stop */}
        {!isRecording ? (
          <button
            onClick={handleStart}
            disabled={!canStart}
            title={
              hasBriefing !== true
                ? "Generate a briefing first"
                : canCaptureInBrowser === false
                ? "Browser doesn't support screen-audio capture — use Chrome or Edge"
                : "Start the live interview companion"
            }
            className="rounded-lg bg-red-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Start recording
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="rounded-lg bg-stone-900 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-stone-800"
          >
            Stop &amp; save
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
        {uploadedToMirror && (
          <span className="ml-3 rounded bg-fuchsia-100 px-2 py-0.5 text-[11px] font-medium text-fuchsia-700">
            ↑ Amphoreus: {uploadedToMirror.filename}
          </span>
        )}
        {savedPath && !trashed && !uploadedToMirror && !showUploadModal && (
          <button
            onClick={() => setShowUploadModal(true)}
            className="ml-2 rounded bg-fuchsia-100 px-2 py-0.5 text-[11px] font-medium text-fuchsia-700 hover:bg-fuchsia-200"
            title="Push this transcript to Amphoreus Supabase so Stelle + the Transcripts tab see it"
          >
            ↑ Upload to Amphoreus
          </button>
        )}
      </div>

      {/* Post-session upload modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-lg rounded-xl bg-white p-6 shadow-xl">
            <h3 className="text-lg font-semibold">Upload to Amphoreus</h3>
            <p className="mt-1 text-xs text-stone-500">
              Pushes this transcript to Amphoreus Supabase. Stelle will see it in <code>{`<slug>/transcripts/`}</code>;
              the Transcripts tab will show it with an <span className="font-medium">interview</span> or
              <span className="font-medium"> sync</span> badge.
            </p>

            <label className="mt-4 block text-xs font-medium text-stone-600">
              Meeting type
            </label>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {([
                { v: "content_interview" as const, label: "Content interview", hint: "Direct interview with the FOC about their content / story / strategy. High-signal." },
                { v: "sync" as const,              label: "Sync",              hint: "Weekly ops check-in, GTM sync, etc. Lower content-generation signal." },
                { v: "other" as const,             label: "Other",             hint: "Anything that doesn't fit interview / sync." },
              ]).map((o) => (
                <button
                  key={o.v}
                  type="button"
                  onClick={() => setUploadSubtype(o.v)}
                  title={o.hint}
                  className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                    uploadSubtype === o.v
                      ? "border-fuchsia-700 bg-fuchsia-700 text-white"
                      : "border-stone-300 bg-white text-stone-600 hover:border-stone-500"
                  }`}
                >
                  {o.label}
                </button>
              ))}
            </div>

            <label className="mt-4 block text-xs font-medium text-stone-600">
              Description <span className="text-stone-400">(visible in the Transcripts tab)</span>
            </label>
            <input
              type="text"
              value={uploadDescription}
              onChange={(e) => setUploadDescription(e.target.value)}
              placeholder={uploadSubtype === "content_interview" ? "e.g. Content interview — Mark — 2026-04-21" : "e.g. Weekly sync — Mark — 2026-04-21"}
              className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-1.5 text-sm focus:border-stone-500 focus:outline-none"
            />

            <div className="mt-3 text-[11px] text-stone-500">
              {transcript.length} segment{transcript.length === 1 ? "" : "s"} · target client:{" "}
              <span className="font-mono text-stone-700">{company}</span>
            </div>

            {uploadError && (
              <div className="mt-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
                {uploadError}
              </div>
            )}

            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                onClick={() => setShowUploadModal(false)}
                disabled={uploadingToMirror}
                className="rounded-lg px-4 py-1.5 text-sm text-stone-500 hover:text-stone-800 disabled:opacity-50"
              >
                Not now
              </button>
              <button
                onClick={() => void handleUploadToAmphoreus()}
                disabled={uploadingToMirror || transcript.length === 0}
                className="rounded-lg bg-fuchsia-700 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-fuchsia-800 disabled:opacity-50"
              >
                {uploadingToMirror ? "Uploading…" : "Upload to Amphoreus"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* No strategic review warning */}
      {!cyreneBrief && !isRunningCyrene && (
        <div className="border-b border-emerald-200 bg-emerald-50 px-6 py-3 text-sm text-emerald-800">
          <span className="font-semibold text-emerald-900">No strategic review yet for {company}. </span>
          Click <strong>Run strategic review</strong> above. Cyrene will study engagement data, ICP
          trends, and warm prospects, then produce a data-backed brief with interview questions,
          DM targets, and content priorities.
        </div>
      )}

      {/* Browser-compat notice. Tribbie now captures audio entirely in
          the browser (getDisplayMedia for tab/system audio + getUserMedia
          for mic), so the only onboarding gate is "are you in a browser
          that supports it" — not BlackHole, not a local install. If the
          feature-detect fails we nudge the user to Chrome/Edge; otherwise
          a tiny hint strip explains what the first-click flow looks like
          so they're not startled by the Chrome screen-picker dialog. */}
      {canCaptureInBrowser === false && (
        <div className="border-b border-amber-300 bg-amber-50 px-6 py-3 text-sm">
          <p className="font-semibold text-amber-900">
            This browser can&rsquo;t capture call audio
          </p>
          <p className="mt-1 text-amber-800">
            Tribbie uses{" "}
            <code className="rounded bg-amber-100 px-1 font-mono text-xs">
              navigator.mediaDevices.getDisplayMedia
            </code>{" "}
            to share the audio from the tab your call is in. That API is
            fully supported in <strong>Chrome</strong> and <strong>Edge</strong> on
            desktop; Safari audio capture support is still partial. Open this
            page in Chrome or Edge and the Start recording button will become
            active.
          </p>
        </div>
      )}

      {canCaptureInBrowser === true && !isRecording && (
        <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-1.5 border-b border-stone-200 bg-stone-50 px-6 py-2 text-xs text-stone-600">
          <p className="flex-1 min-w-[300px]">
            <span className="font-medium text-stone-700">How recording works:</span>{" "}
            click <strong>Start recording</strong>
            {captureMic ? ", allow microphone access, " : ", "}
            then pick the tab your Zoom/Meet/Slack call is in and tick{" "}
            <strong>Share tab audio</strong>.{" "}
            {captureMic
              ? "Mic + remote voices mix locally, stream to the cloud for transcription,"
              : "Remote voices stream to the cloud for transcription (your mic is not captured),"}{" "}
            and the transcript uploads to Amphoreus when you stop.
          </p>
          <label className="flex cursor-pointer items-center gap-2 whitespace-nowrap rounded border border-stone-200 bg-white px-2 py-1 text-stone-700 hover:bg-stone-100">
            <input
              type="checkbox"
              className="h-3.5 w-3.5 accent-stone-700"
              checked={captureMic}
              onChange={(e) => toggleCaptureMic(e.target.checked)}
            />
            <span>
              Capture my mic too
              <span className="ml-1 text-stone-400">
                {captureMic ? "(headphones recommended)" : "(interviewee-only transcript)"}
              </span>
            </span>
          </label>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="border-b border-red-200 bg-red-50 px-6 py-3 text-sm text-red-700">
          <span className="font-medium">Error: </span>
          {error}
        </div>
      )}

      {/* Briefing generation log (collapsible while streaming) */}
      {showBriefingLog && (
        <div className="flex max-h-64 flex-col border-b border-stone-200 bg-stone-950 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-stone-800 px-4 py-1.5 text-stone-400">
            <span>Aglaea progress</span>
            <button
              onClick={() => setShowBriefingLog(false)}
              className="text-stone-500 hover:text-stone-300"
            >
              ×
            </button>
          </div>
          <div ref={briefingLogRef} className="flex-1 space-y-0.5 overflow-auto p-3">
            {briefingLog.map((line, i) => (
              <div key={i} className={getBriefingLineColor(line.type)}>
                <span className="mr-2 text-stone-600">
                  {new Date(line.timestamp).toLocaleTimeString()}
                </span>
                <span className="mr-2 text-stone-500">[{line.type}]</span>
                <span className="whitespace-pre-wrap">{line.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cyrene strategic review log (collapsible while streaming) */}
      {showCyreneLog && (
        <div className="flex max-h-64 flex-col border-b border-stone-200 bg-stone-950 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-stone-800 px-4 py-1.5 text-stone-400">
            <span>Cyrene strategic review progress</span>
            <button
              onClick={() => setShowCyreneLog(false)}
              className="text-stone-500 hover:text-stone-300"
            >
              &times;
            </button>
          </div>
          <div className="flex-1 space-y-0.5 overflow-auto p-3">
            {cyreneLog.map((line, i) => (
              <div key={i} className={line.type === "error" ? "text-red-400" : "text-stone-300"}>
                <span className="mr-2 text-stone-600">
                  {new Date(line.timestamp).toLocaleTimeString()}
                </span>
                <span className="mr-2 text-stone-500">[{line.type}]</span>
                <span className="whitespace-pre-wrap">{line.text.slice(0, 300)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cyrene strategic brief display (when a brief exists) */}
      {showCyreneBrief && cyreneBrief && !showCyreneLog && (
        <div className="border-b border-emerald-200 bg-emerald-50/50 px-6 py-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-emerald-900">
              Cyrene Strategic Brief
              {cyreneBrief._computed_at && (
                <span className="ml-2 font-normal text-emerald-600">
                  ({new Date(cyreneBrief._computed_at).toLocaleDateString()})
                </span>
              )}
            </h3>
            <div className="flex items-center gap-3">
              <button
                onClick={handleDownloadBrief}
                className="text-xs text-emerald-700 hover:text-emerald-900 underline-offset-2 hover:underline"
                title="Download as HTML — open in any browser, save-as-PDF if you want"
              >
                Download
              </button>
              <button
                onClick={() => setShowCyreneBrief(false)}
                className="text-xs text-emerald-500 hover:text-emerald-700"
              >
                Hide
              </button>
            </div>
          </div>

          <div className="mt-3 flex flex-col gap-4 text-xs">
            {/* Current Strategy Diagnosis — the spine of the brief */}
            {cyreneBrief.current_strategy_diagnosis && (
              <div className="rounded-lg border border-emerald-300 bg-white p-3">
                <h4 className="mb-2 text-sm font-semibold text-emerald-900">
                  Current strategy — diagnosis
                </h4>
                {cyreneBrief.current_strategy_diagnosis.what_is_working?.length > 0 && (
                  <div className="mb-3">
                    <h5 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-emerald-700">
                      What's working ({cyreneBrief.current_strategy_diagnosis.what_is_working.length})
                    </h5>
                    <ul className="space-y-1.5 text-stone-700">
                      {cyreneBrief.current_strategy_diagnosis.what_is_working.map((w: any, i: number) => (
                        <li key={i}>
                          <div className="font-medium text-emerald-900">{w.pattern}</div>
                          {w.evidence && <p className="mt-0.5 text-stone-600">{w.evidence}</p>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {cyreneBrief.current_strategy_diagnosis.what_is_broken?.length > 0 && (
                  <div className="mb-3">
                    <h5 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-red-700">
                      What's broken ({cyreneBrief.current_strategy_diagnosis.what_is_broken.length})
                    </h5>
                    <ul className="space-y-1.5 text-stone-700">
                      {cyreneBrief.current_strategy_diagnosis.what_is_broken.map((b: any, i: number) => (
                        <li key={i}>
                          <div className="font-medium text-red-900">{b.pattern}</div>
                          {b.evidence && <p className="mt-0.5 text-stone-600">{b.evidence}</p>}
                          {b.consequence && (
                            <p className="mt-0.5 text-stone-500 italic">Consequence: {b.consequence}</p>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {cyreneBrief.current_strategy_diagnosis.blind_spots?.length > 0 && (
                  <div>
                    <h5 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-amber-700">
                      Blind spots ({cyreneBrief.current_strategy_diagnosis.blind_spots.length})
                    </h5>
                    <ul className="space-y-1.5 text-stone-700">
                      {cyreneBrief.current_strategy_diagnosis.blind_spots.map((s: any, i: number) => (
                        <li key={i}>
                          <div className="font-medium text-amber-900">{s.observation}</div>
                          {s.evidence && <p className="mt-0.5 text-stone-600">{s.evidence}</p>}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Strategic Themes — 4-8 week directions for the public voice */}
            {cyreneBrief.strategic_themes?.length > 0 && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">
                  Strategic Themes ({cyreneBrief.strategic_themes.length})
                </h4>
                <ul className="space-y-2 text-stone-700">
                  {cyreneBrief.strategic_themes.map((t: any, i: number) => (
                    <li key={i}>
                      <div className="font-medium text-emerald-900">
                        {t.theme}
                        {t.addresses && (
                          <span className="ml-2 rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-[10px] font-normal text-emerald-700">
                            ↳ {t.addresses}
                          </span>
                        )}
                      </div>
                      {t.evidence && (
                        <p className="mt-0.5 text-stone-600">{t.evidence}</p>
                      )}
                      {t.arc && (
                        <p className="mt-0.5 text-stone-500 italic">Arc: {t.arc}</p>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Topics to Probe — Tribbie's interview menu */}
            {cyreneBrief.topics_to_probe?.length > 0 && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">
                  Topics to Probe ({cyreneBrief.topics_to_probe.length})
                </h4>
                <ol className="list-decimal space-y-2 pl-4 text-stone-700">
                  {cyreneBrief.topics_to_probe.map((p: any, i: number) => (
                    <li key={i}>
                      <div className="font-medium text-emerald-900">{p.thread}</div>
                      {p.why && (
                        <p className="mt-0.5 text-stone-600">{p.why}</p>
                      )}
                      {p.suggested_entry_point && (
                        <p className="mt-0.5 text-stone-500 italic">Entry: {p.suggested_entry_point}</p>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {/* DM Targets */}
            {cyreneBrief.dm_targets?.length > 0 && (
              <div className="rounded-lg border border-emerald-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-emerald-800">
                  DM Targets ({cyreneBrief.dm_targets.length})
                </h4>
                <div className="space-y-2">
                  {cyreneBrief.dm_targets.map((t: any, i: number) => (
                    <div key={i} className="text-stone-700">
                      <span className="font-medium">{t.name}</span>
                      {t.company && <span className="text-stone-500"> @ {t.company}</span>}
                      {t.suggested_angle && (
                        <p className="mt-0.5 text-stone-500 italic">{t.suggested_angle}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Topics Exhausted — worn-out patterns to steer away from */}
            {cyreneBrief.topics_exhausted?.length > 0 && (
              <div className="rounded-lg border border-red-200 bg-white p-3">
                <h4 className="mb-1.5 font-semibold text-red-800">
                  Topics Exhausted ({cyreneBrief.topics_exhausted.length})
                </h4>
                <ul className="space-y-1.5 text-stone-700">
                  {cyreneBrief.topics_exhausted.map((e: any, i: number) => (
                    <li key={i}>
                      <div className="font-medium text-red-900">{e.pattern}</div>
                      {e.evidence && (
                        <p className="mt-0.5 text-stone-600">{e.evidence}</p>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Prose — strategic narrative (collapsed-ish) */}
            {cyreneBrief.prose && (
              <details className="rounded-lg border border-emerald-200 bg-white p-3">
                <summary className="cursor-pointer font-semibold text-emerald-800">
                  Strategic narrative
                </summary>
                <p className="mt-2 whitespace-pre-wrap leading-relaxed text-stone-700">
                  {cyreneBrief.prose}
                </p>
              </details>
            )}
          </div>

          {/* Meta footer */}
          <div className="mt-3 flex gap-4 text-xs text-emerald-600">
            {cyreneBrief._turns_used && <span>{cyreneBrief._turns_used} turns</span>}
            {cyreneBrief._cost_usd && <span>${cyreneBrief._cost_usd}</span>}
            {cyreneBrief.next_run_trigger?.condition && (
              <span>Next: {cyreneBrief.next_run_trigger.condition}</span>
            )}
          </div>
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
          {/* Topics to Probe (from Cyrene brief) — Tribbie's topic menu for this call */}
          <div className="flex h-1/2 flex-col border-b border-stone-200">
            <div className="flex shrink-0 items-center justify-between border-b border-stone-100 bg-stone-50 px-4 py-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-stone-500">
                Topics to Probe
              </span>
              {cyreneBrief?.topics_to_probe && (
                <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-semibold text-emerald-600">
                  {cyreneBrief.topics_to_probe.length}
                </span>
              )}
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {cyreneBrief?.topics_to_probe?.length > 0 ? (
                <ol className="list-decimal space-y-3 pl-4 text-sm text-stone-700">
                  {cyreneBrief.topics_to_probe.map((p: any, i: number) => (
                    <li key={i} className="leading-relaxed">
                      <div className="font-medium text-stone-900">{p.thread}</div>
                      {p.why && (
                        <p className="mt-0.5 text-xs text-stone-500">{p.why}</p>
                      )}
                      {p.suggested_entry_point && (
                        <p className="mt-0.5 text-xs italic text-stone-500">
                          Entry: {p.suggested_entry_point}
                        </p>
                      )}
                    </li>
                  ))}
                </ol>
              ) : briefingContent ? (
                <div className="prose prose-sm prose-stone max-w-none text-xs [&_h1]:text-sm [&_h2]:text-xs [&_h3]:text-xs [&_li]:my-0.5 [&_p]:my-1">
                  <ReactMarkdown>{briefingContent}</ReactMarkdown>
                </div>
              ) : (
                <p className="text-sm text-stone-400">
                  Run a strategic review to generate data-backed topics to probe.
                </p>
              )}
            </div>
          </div>

          {/* Suggestions panel */}
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

function extractBriefingText(data: Record<string, unknown>): string {
  const d = data.data as Record<string, unknown> | undefined;
  if (!d) return JSON.stringify(data);
  if (data.type === "done") return ((d.output as string) || "").slice(0, 200) || "Briefing complete.";
  if (data.type === "error") return (d.message as string) || "Unknown error";
  if (data.type === "tool_call") {
    const args = d.arguments as Record<string, unknown> | undefined;
    return `${d.name}(${(args?.summary as string) || JSON.stringify(d.arguments) || ""})`;
  }
  if (data.type === "tool_result") return `${d.name} -> ${((d.result as string) || "").slice(0, 200)}`;
  return (d.text as string) || (d.message as string) || (d.name as string) || JSON.stringify(d);
}

function getBriefingLineColor(type: string): string {
  switch (type) {
    case "thinking":
      return "text-blue-400";
    case "tool_call":
      return "text-amber-400";
    case "tool_result":
      return "text-emerald-400";
    case "text_delta":
      return "text-stone-200";
    case "compaction":
      return "text-purple-400";
    case "error":
      return "text-red-400";
    case "done":
      return "text-green-400";
    case "status":
      return "text-cyan-400";
    default:
      return "text-stone-400";
  }
}
