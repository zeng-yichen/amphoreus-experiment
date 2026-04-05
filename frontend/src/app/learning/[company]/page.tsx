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

function ReadinessBar({ label, current, target, active }: { label: string; current: number; target: number; active: boolean }) {
  const pct = Math.min(100, (current / Math.max(target, 1)) * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-stone-400">{label}</span>
        <span className={active ? "text-emerald-400" : "text-stone-500"}>
          {active ? "ACTIVE" : `${current}/${target}`}
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-stone-800">
        <div
          className={`h-1.5 rounded-full transition-all ${active ? "bg-emerald-500" : "bg-stone-600"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
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
  const temporal = data.temporal;
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

        {/* Adaptive readiness */}
        <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
          <p className="text-sm font-medium text-stone-300">Adaptive System Readiness</p>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <ReadinessBar
              label="Permansor Weights"
              current={rd.permansor_dims}
              target={10}
              active={rd.permansor_weights_ready}
            />
            <ReadinessBar
              label="Constitutional Learning"
              current={rd.permansor_dims}
              target={15}
              active={rd.constitutional_ready}
            />
            <ReadinessBar
              label="Freeform Critic"
              current={obs.scored}
              target={10}
              active={rd.freeform_critic_active}
            />
            <ReadinessBar
              label="Emergent Dimensions"
              current={rd.permansor_dims}
              target={40}
              active={rd.emergent_dims_ready}
            />
            <ReadinessBar
              label="Continuous LOLA"
              current={lola?.content_points || 0}
              target={10}
              active={rd.continuous_lola_active}
            />
            <div className="flex items-center gap-2 rounded bg-stone-800/50 px-3 py-2">
              <span className="text-xs text-stone-500">Dimension set:</span>
              <span className={`text-xs font-medium ${rd.current_dimension_set === "fixed_v1" ? "text-stone-400" : "text-emerald-400"}`}>
                {rd.current_dimension_set}
              </span>
            </div>
          </div>
        </div>

        {/* LOLA */}
        {lola && (
          <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-stone-300">
                Content Direction (LOLA)
              </p>
              <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${lola.using_continuous ? "bg-emerald-950 text-emerald-400" : "bg-stone-800 text-stone-400"}`}>
                {lola.using_continuous ? "Continuous Field" : "Arm-Based"}
              </span>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
              <div>
                <span className="text-stone-500">Pulls:</span>{" "}
                <span className="text-stone-200">{lola.total_pulls}</span>
              </div>
              <div>
                <span className="text-stone-500">Content Points:</span>{" "}
                <span className="text-stone-200">{lola.content_points}</span>
              </div>
              <div>
                <span className="text-stone-500">Exploration:</span>{" "}
                <span className="text-stone-200">{(lola.exploration_rate * 100).toFixed(0)}%</span>
              </div>
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

        {/* Temporal */}
        {temporal && (
          <div className="mt-6 rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-sm font-medium text-stone-300">Scheduling Intelligence</p>
            <div className="mt-3 flex flex-wrap gap-4 text-sm">
              <div>
                <span className="text-stone-500">Best days:</span>{" "}
                <span className="text-stone-200">{temporal.best_days?.join(", ")}</span>
              </div>
              <div>
                <span className="text-stone-500">Best hours:</span>{" "}
                <span className="text-stone-200">{temporal.best_hours?.map((h: number) => `${h}:00`).join(", ")}</span>
              </div>
              {temporal.cooldown_hours && (
                <div>
                  <span className="text-stone-500">Cooldown:</span>{" "}
                  <span className="text-stone-200">{Math.round(temporal.cooldown_hours)}h</span>
                </div>
              )}
            </div>
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
