"use client";

import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import Link from "next/link";
import { learningApi } from "@/lib/api";

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="rounded-lg border border-stone-800 bg-stone-900 p-4">
      <p className="text-xs font-medium text-stone-500">{label}</p>
      <p className="mt-1 text-xl font-semibold">{value}</p>
      {sub && <p className="text-xs text-stone-600">{sub}</p>}
    </div>
  );
}

function ReadinessIndicator({ label, active }: { label: string; active: boolean }) {
  return (
    <div className="flex items-center gap-2 rounded bg-stone-800/50 px-3 py-2">
      <div className={`h-2 w-2 rounded-full ${active ? "bg-emerald-500" : "bg-stone-600"}`} />
      <span className="text-xs text-stone-400">{label}</span>
      <span className={`ml-auto text-xs font-medium ${active ? "text-emerald-400" : "text-stone-500"}`}>
        {active ? "ACTIVE" : "WAITING"}
      </span>
    </div>
  );
}

function ConfidenceBadge({ confidence }: { confidence: string }) {
  const colors: Record<string, string> = {
    strong: "bg-emerald-950 text-emerald-400",
    suggestive: "bg-amber-950 text-amber-400",
    weak: "bg-stone-800 text-stone-400",
  };
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${colors[confidence] || colors.weak}`}>
      {confidence}
    </span>
  );
}

export default function ClientLearningDetail() {
  const params = useParams();
  const company = params.company as string;
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    learningApi
      .getClient(company)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [company]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-stone-950">
        <p className="text-stone-400">Loading...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-stone-950">
        <p className="text-stone-400">No data for {company}</p>
      </div>
    );
  }

  const obs = data.observations || {};
  const rs = data.reward_stats || {};
  const eng = data.engagement || {};
  const cad = data.cadence || {};
  const lola = data.lola;
  const rd = data.readiness || {};
  const tags = data.tags || {};
  const analyst = data.analyst;
  const directives = data.directives;
  const icp = data.icp;

  return (
    <div className="min-h-screen bg-stone-950 text-stone-100">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">{company}</h1>
            <p className="mt-1 text-sm text-stone-400">Learning Intelligence</p>
          </div>
          <Link href="/learning" className="text-sm text-stone-500 hover:text-stone-300">
            ← All Clients
          </Link>
        </div>

        {/* Summary stats */}
        <div className="mt-8 grid grid-cols-5 gap-3">
          <StatCard label="Observations" value={obs.total} sub={`${obs.scored} scored (${obs.pct_scored}%)`} />
          <StatCard label="Avg Impressions" value={eng.avg_impressions?.toLocaleString() || "0"} />
          <StatCard label="Avg Reactions" value={eng.avg_reactions || 0} />
          <StatCard label="Cadence" value={`${cad.avg_days}d`} sub={`${cad.posts_last_7d} last 7d`} />
          <StatCard
            label="Reward"
            value={`${rs.mean >= 0 ? "+" : ""}${rs.mean}`}
            sub={`σ=${rs.std} [${rs.min}, ${rs.max}]`}
          />
        </div>

        {/* Reward sparkline */}
        {data.reward_sparkline && (
          <div className="mt-4 rounded-lg border border-stone-800 bg-stone-900 p-4">
            <p className="text-xs font-medium text-stone-500">Reward Trend (last 12 posts)</p>
            <p className="mt-2 font-mono text-2xl tracking-widest text-stone-200">{data.reward_sparkline}</p>
          </div>
        )}

        {/* System readiness */}
        <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
          <p className="text-sm font-medium text-stone-300">System Readiness</p>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <ReadinessIndicator label="Observation Tagger" active={rd.observation_tagger_active} />
            <ReadinessIndicator label="Analyst Agent" active={rd.analyst_active} />
            <ReadinessIndicator label="Learned Directives" active={rd.directives_active} />
            <ReadinessIndicator label="Freeform Critic" active={rd.freeform_critic_active} />
            <ReadinessIndicator label="Cyrene Weights" active={rd.cyrene_weights_ready} />
          </div>
        </div>

        {/* Analyst Findings */}
        {analyst && analyst.latest_findings?.length > 0 && (
          <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-stone-300">Analyst Findings</p>
              {analyst.last_run && (
                <span className="text-[10px] text-stone-500">
                  {analyst.last_run.timestamp?.slice(0, 10)} · {analyst.last_run.tool_calls} tools ·{" "}
                  {Math.round(analyst.last_run.elapsed_seconds)}s · {analyst.total_runs} runs total
                </span>
              )}
            </div>
            <div className="mt-4 space-y-3">
              {analyst.latest_findings.map((f: any, i: number) => (
                <div key={i} className="border-l-2 border-stone-700 pl-3">
                  <div className="flex items-start gap-2">
                    <ConfidenceBadge confidence={f.confidence} />
                    <p className="text-sm text-stone-200">{f.claim}</p>
                  </div>
                  {f.evidence && (
                    <p className="mt-1 text-xs text-stone-500">{f.evidence}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Observation Tags */}
        {tags.tagged_count > 0 && (
          <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-sm font-medium text-stone-300">Content Distribution</p>
            <p className="mt-1 text-xs text-stone-500">{tags.tagged_count} posts tagged</p>
            <div className="mt-4 grid grid-cols-2 gap-6">
              <div>
                <p className="text-xs text-stone-500">Topics</p>
                <div className="mt-2 space-y-1">
                  {Object.entries(tags.topics || {}).map(([topic, count]: [string, any]) => (
                    <div key={topic} className="flex items-center justify-between text-sm">
                      <span className="text-stone-300">{topic}</span>
                      <span className="text-stone-500">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-xs text-stone-500">Formats</p>
                <div className="mt-2 space-y-1">
                  {Object.entries(tags.formats || {}).map(([fmt, count]: [string, any]) => (
                    <div key={fmt} className="flex items-center justify-between text-sm">
                      <span className="text-stone-300">{fmt}</span>
                      <span className="text-stone-500">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Learned Directives */}
        {directives && directives.rules?.length > 0 && (
          <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-sm font-medium text-stone-300">Learned Writing Rules</p>
            <p className="mt-1 text-xs text-stone-500">{directives.count} rules from {
              [...new Set(directives.rules.map((r: any) => r.source))].join(", ")
            }</p>
            <div className="mt-4 space-y-2">
              {directives.rules.map((r: any, i: number) => (
                <div key={i} className="flex items-start gap-2 text-sm">
                  <span className={`mt-0.5 shrink-0 rounded px-1 py-0.5 text-[10px] font-medium ${
                    r.priority === "high"
                      ? "bg-red-950 text-red-400"
                      : "bg-stone-800 text-stone-400"
                  }`}>
                    {r.priority}
                  </span>
                  <span className="text-stone-300">{r.directive}</span>
                  {r.efficacy !== "untested" && (
                    <span className={`shrink-0 text-[10px] ${
                      r.efficacy === "validated" ? "text-emerald-400" :
                      r.efficacy === "counterproductive" ? "text-red-400" : "text-stone-500"
                    }`}>
                      {r.efficacy}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* LOLA */}
        {lola && (
          <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-stone-300">Content Direction (LOLA)</p>
              <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${
                lola.using_continuous ? "bg-emerald-950 text-emerald-400" : "bg-stone-800 text-stone-400"
              }`}>
                {lola.using_continuous ? "Continuous Field" : "Arm-Based"}
              </span>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
              <div><span className="text-stone-500">Pulls:</span> <span className="text-stone-200">{lola.total_pulls}</span></div>
              <div><span className="text-stone-500">Content Points:</span> <span className="text-stone-200">{lola.content_points}</span></div>
              <div><span className="text-stone-500">Exploration:</span> <span className="text-stone-200">{(lola.exploration_rate * 100).toFixed(0)}%</span></div>
            </div>
            {lola.top_arms?.length > 0 && (
              <div className="mt-4">
                <p className="text-xs text-stone-500">Top directions</p>
                <div className="mt-2 space-y-1">
                  {lola.top_arms.map((a: any, i: number) => (
                    <div key={i} className="flex items-center justify-between text-sm">
                      <span className="text-stone-300">{a.label}</span>
                      <span>
                        <span className={a.avg_reward >= 0 ? "text-emerald-400" : "text-red-400"}>
                          {a.avg_reward > 0 ? "+" : ""}{a.avg_reward}
                        </span>
                        <span className="ml-2 text-stone-600">{a.pulls}p</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ICP */}
        {icp && (
          <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-sm font-medium text-stone-300">ICP Definition</p>
            <p className="mt-2 text-sm text-stone-400">{icp.description}</p>
            {icp.anti_description && (
              <p className="mt-2 text-xs text-stone-600">Anti-ICP: {icp.anti_description}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
