"use client";

import Link from "next/link";
import { useState } from "react";
import { desktopApi } from "@/lib/api";

export default function HomePage() {
  const [company, setCompany] = useState("");
  const [guiStatus, setGuiStatus] = useState<string | null>(null);

  const slug = company.trim().toLowerCase().replace(/\s+/g, "-") || "default";

  const workflows = [
    { title: "Ghostwriter", href: `/ghostwriter/${slug}`, description: "Generate LinkedIn posts via Stelle" },
    { title: "Interview Briefing", href: `/briefings/${slug}`, description: "Generate pre-call questions via Aglaea" },
    { title: "Interview Companion", href: `/interview/${slug}`, description: "Live transcript + follow-up suggestions via Tribbie" },
    { title: "Content Strategy", href: `/strategy/${slug}`, description: "Generate or view content strategy via Screwllum" },
    { title: "Customer Success", href: "/cs", description: "Client health and analytics" },
  ];

  async function launchClassicGui() {
    setGuiStatus("launching...");
    try {
      const res = await desktopApi.launch();
      setGuiStatus(res.status === "already_running" ? "already open" : "launched");
      setTimeout(() => setGuiStatus(null), 3000);
    } catch {
      setGuiStatus("failed — is the backend running?");
      setTimeout(() => setGuiStatus(null), 4000);
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">Amphoreus</h1>
      <p className="mt-2 text-stone-500">AI content operations platform</p>

      <div className="mt-8 flex items-end gap-4">
        <div className="flex-1">
          <label htmlFor="company" className="block text-sm font-medium text-stone-700">
            Company keyword
          </label>
          <input
            id="company"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="e.g. hume-andrew, innovocommerce"
            className="mt-1 w-full rounded-lg border border-stone-300 px-3 py-2 text-sm shadow-sm focus:border-stone-500 focus:outline-none focus:ring-1 focus:ring-stone-500"
          />
        </div>
        <p className="pb-2 text-xs text-stone-400">
          This maps to <code className="rounded bg-stone-100 px-1">memory/{slug}/</code>
        </p>
      </div>

      <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {workflows.map((w) => (
          <Link
            key={w.title}
            href={w.href}
            className="group rounded-xl border border-stone-200 bg-white p-6 shadow-sm transition-all hover:border-stone-300 hover:shadow-md"
          >
            <h2 className="text-lg font-medium text-stone-900 group-hover:text-stone-700">
              {w.title}
            </h2>
            <p className="mt-1 text-sm text-stone-500">{w.description}</p>
          </Link>
        ))}

        <button
          onClick={launchClassicGui}
          className="group rounded-xl border border-dashed border-stone-300 bg-stone-50 p-6 text-left shadow-sm transition-all hover:border-stone-400 hover:bg-white hover:shadow-md"
        >
          <h2 className="text-lg font-medium text-stone-900 group-hover:text-stone-700">
            Classic Desktop GUI
          </h2>
          <p className="mt-1 text-sm text-stone-500">
            {guiStatus ?? "Launch the original Tkinter interface"}
          </p>
        </button>
      </div>
    </div>
  );
}
