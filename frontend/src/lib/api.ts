/**
 * API client for the Amphoreus backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getAuthHeaders(): Promise<Record<string, string>> {
  // TODO: Get Supabase session token
  const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { ...headers, ...options.headers },
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }
  return res.json();
}

// --- Shared SSE stream helper ---

async function* streamSSE(url: string) {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}${url}`, { headers });
  if (!res.ok || !res.body) return;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.replace(/^data: /, "").trim();
      if (line) {
        try { yield JSON.parse(line); } catch { /* skip */ }
      }
    }
  }
}

// --- Ghostwriter ---

export const ghostwriterApi = {
  generate: (company: string, prompt?: string, model?: string) =>
    apiFetch<{ job_id: string; status: string }>("/api/ghostwriter/generate", {
      method: "POST",
      body: JSON.stringify({ company, prompt, model }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/ghostwriter/stream/${jobId}`),

  getJob: (jobId: string) =>
    apiFetch<{ job_id: string; status: string; output?: string; error?: string }>(
      `/api/ghostwriter/jobs/${jobId}`
    ),

  getRuns: (company: string, limit = 20) =>
    apiFetch<{ runs: any[] }>(`/api/ghostwriter/${company}/runs?limit=${limit}`),

  getRunEvents: (runId: string) =>
    apiFetch<{ run: any; events: any[] }>(`/api/ghostwriter/runs/${runId}/events`),

  rollback: (company: string, runId: string) =>
    apiFetch(`/api/ghostwriter/${company}/rollback/${runId}`, { method: "POST" }),

  getFiles: (company: string, path = "") =>
    apiFetch<{ files: any[] }>(`/api/ghostwriter/sandbox/${company}/files?path=${path}`),

  submitFeedback: (company: string, original: string, revised: string) =>
    apiFetch("/api/ghostwriter/feedback", {
      method: "POST",
      body: JSON.stringify({ company, original, revised }),
    }),

  inlineEdit: (company: string, postText: string, instruction: string) =>
    apiFetch<{ result: string }>("/api/ghostwriter/inline-edit", {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, instruction }),
    }),
};

// --- Briefings ---

export const briefingsApi = {
  generate: (clientName: string, company: string) =>
    apiFetch<{ job_id: string }>("/api/briefings/generate", {
      method: "POST",
      body: JSON.stringify({ client_name: clientName, company }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/briefings/stream/${jobId}`),

  check: (company: string) =>
    apiFetch<{ exists: boolean }>(`/api/briefings/check/${company}`),

  get: (company: string) =>
    apiFetch<{ content: string }>(`/api/briefings/content/${company}`),
};

// --- Strategy ---

export const strategyApi = {
  generate: (company: string, prompt?: string) =>
    apiFetch<{ job_id: string }>("/api/strategy/generate", {
      method: "POST",
      body: JSON.stringify({ company, prompt }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/strategy/stream/${jobId}`),

  getCurrent: (company: string) =>
    apiFetch<{ strategy: string | null; path?: string }>(`/api/strategy/${company}`),
};

// --- Posts ---

export const postsApi = {
  list: (company?: string, limit = 50) =>
    apiFetch<{ posts: any[] }>(`/api/posts?${company ? `company=${company}&` : ""}limit=${limit}`),

  create: (company: string, content: string, title?: string) =>
    apiFetch("/api/posts", {
      method: "POST",
      body: JSON.stringify({ company, content, title }),
    }),

  update: (postId: string, company: string, content: string, status?: string) =>
    apiFetch(`/api/posts/${postId}`, {
      method: "PATCH",
      body: JSON.stringify({ company, content, status }),
    }),

  delete: (postId: string) =>
    apiFetch(`/api/posts/${postId}`, { method: "DELETE" }),

  rewrite: (postId: string, company: string, postText: string, styleInstruction?: string) =>
    apiFetch<{ result: any }>(`/api/posts/${postId}/rewrite`, {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, style_instruction: styleInstruction }),
    }),

  factCheck: (postId: string, company: string, postText: string) =>
    apiFetch<{ report: string }>(`/api/posts/${postId}/fact-check`, {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText }),
    }),

  push: (company: string, content: string) =>
    apiFetch<{ success: boolean; result: string }>("/api/posts/push", {
      method: "POST",
      body: JSON.stringify({ company, content }),
    }),
};

// --- Images ---

export const imagesApi = {
  generate: (company: string, postText: string, model?: string) =>
    apiFetch<{ job_id: string; status: string }>("/api/images/generate", {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, model }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/images/stream/${jobId}`),

  list: (company: string, limit = 50) =>
    apiFetch<{ images: any[] }>(`/api/images/${company}?limit=${limit}`),

  getUrl: (company: string, imageId: string) =>
    `${API_BASE}/api/images/${company}/${imageId}`,
};

// --- Research ---

export const researchApi = {
  web: (highlightedText: string, query: string) =>
    apiFetch<{ result: string }>("/api/research/web", {
      method: "POST",
      body: JSON.stringify({ highlighted_text: highlightedText, query }),
    }),

  documents: (company: string, question: string, draftText?: string) =>
    apiFetch<{ result: string }>("/api/research/documents", {
      method: "POST",
      body: JSON.stringify({ company, question, draft_text: draftText }),
    }),

  source: (snippet: string, company: string) =>
    apiFetch<{ result: string }>("/api/research/source", {
      method: "POST",
      body: JSON.stringify({ snippet, company }),
    }),

  abm: (company: string) =>
    apiFetch<{ result: string }>("/api/research/abm", {
      method: "POST",
      body: JSON.stringify({ company }),
    }),
};

// --- Desktop GUI ---

export const desktopApi = {
  launch: () =>
    apiFetch<{ status: string; pid: number }>("/api/desktop/launch", { method: "POST" }),

  status: () =>
    apiFetch<{ running: boolean; pid: number | null }>("/api/desktop/status"),
};

// --- Auth ---

export const authApi = {
  getPermissions: () => apiFetch<{ user_id: string; role: string }>("/api/auth/permissions"),
  getProfile: () => apiFetch<{ profile: any }>("/api/auth/profile"),
};

// --- CS Dashboard ---

export const csApi = {
  listClients: () => apiFetch<{ clients: any[] }>("/api/cs/clients"),
  getClient: (clientId: string) => apiFetch<{ user: any; posts: any[] }>(`/api/cs/clients/${clientId}`),
};

// --- Interview Companion (Tribbie) ---

export const interviewApi = {
  listDevices: () =>
    apiFetch<{ devices: any[]; has_blackhole: boolean; error?: string }>("/api/interview/devices"),

  start: (company: string, clientName?: string) =>
    apiFetch<{ job_id: string; status: string }>("/api/interview/start", {
      method: "POST",
      body: JSON.stringify({ company, client_name: clientName }),
    }),

  stop: (jobId: string, company: string) =>
    apiFetch<{ status: string; job_id: string }>("/api/interview/stop", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId, company }),
    }),

  streamJob: (jobId: string) => streamSSE(`/api/interview/stream/${jobId}`),

  trashTranscript: (path: string) =>
    apiFetch<{ status: string; destination: string }>("/api/interview/trash-transcript", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),
};
