"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { learningApi } from "@/lib/api";

interface LearningClient {
  slug: string;
  scored: number;
  total: number;
}

export default function LearningDashboard() {
  const [clients, setClients] = useState<LearningClient[]>([]);
  const [crossClient, setCrossClient] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      learningApi.listClients(),
      learningApi.getCrossClient(),
    ])
      .then(([clientData, ccData]) => {
        setClients(clientData.clients.sort((a: LearningClient, b: LearningClient) => b.scored - a.scored));
        setCrossClient(ccData);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-stone-950">
        <p className="text-stone-400">Loading learning data...</p>
      </div>
    );
  }

  const totalObs = clients.reduce((s, c) => s + c.total, 0);
  const totalScored = clients.reduce((s, c) => s + c.scored, 0);

  return (
    <div className="min-h-screen bg-stone-950 text-stone-100">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Learning Intelligence</h1>
            <p className="mt-1 text-sm text-stone-400">
              Adaptive system health and content learning progress
            </p>
          </div>
          <Link href="/home" className="text-sm text-stone-500 hover:text-stone-300">
            ← Home
          </Link>
        </div>

        {/* Cross-client summary */}
        <div className="mt-8 grid grid-cols-4 gap-4">
          <div className="rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-xs font-medium text-stone-500">Total Observations</p>
            <p className="mt-1 text-2xl font-semibold">{totalObs}</p>
            <p className="text-xs text-stone-600">{totalScored} scored</p>
          </div>
          <div className="rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-xs font-medium text-stone-500">Active Clients</p>
            <p className="mt-1 text-2xl font-semibold">{clients.length}</p>
          </div>
          <div className="rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-xs font-medium text-stone-500">Hook Library</p>
            <p className="mt-1 text-2xl font-semibold">{crossClient?.hook_library_size || 0}</p>
            <p className="text-xs text-stone-600">exemplars</p>
          </div>
          <div className="rounded-lg border border-stone-800 bg-stone-900 p-5">
            <p className="text-xs font-medium text-stone-500">Universal Patterns</p>
            <p className="mt-1 text-2xl font-semibold">{crossClient?.universal_patterns || 0}</p>
            <p className="text-xs text-stone-600">cross-client</p>
          </div>
        </div>

        {/* Top/bottom arms */}
        {crossClient?.top_arms?.length > 0 && (
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="rounded-lg border border-stone-800 bg-stone-900 p-5">
              <p className="text-xs font-medium text-emerald-400">Top Performing Arms</p>
              <div className="mt-3 space-y-2">
                {crossClient.top_arms.slice(0, 5).map((a: any, i: number) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="text-stone-300">{a.label}</span>
                    <span className="text-stone-500">
                      <span className="text-emerald-400">{a.avg_reward > 0 ? "+" : ""}{a.avg_reward}</span>
                      {" · "}{a.company}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-lg border border-stone-800 bg-stone-900 p-5">
              <p className="text-xs font-medium text-red-400">Lowest Performing Arms</p>
              <div className="mt-3 space-y-2">
                {crossClient.bottom_arms.slice(0, 5).map((a: any, i: number) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="text-stone-300">{a.label}</span>
                    <span className="text-stone-500">
                      <span className="text-red-400">{a.avg_reward > 0 ? "+" : ""}{a.avg_reward}</span>
                      {" · "}{a.company}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Client list */}
        <h2 className="mt-10 text-lg font-medium">Clients</h2>
        <div className="mt-4 space-y-2">
          {clients.map((c) => (
            <Link
              key={c.slug}
              href={`/learning/${c.slug}`}
              className="flex items-center justify-between rounded-lg border border-stone-800 bg-stone-900 p-4 transition-all hover:border-stone-700 hover:bg-stone-800/80"
            >
              <div>
                <span className="font-medium text-stone-200">{c.slug}</span>
              </div>
              <div className="flex items-center gap-6 text-sm text-stone-400">
                <span>{c.scored} scored</span>
                <span className="text-stone-600">{c.total} total</span>
                <span className="text-stone-600">→</span>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
