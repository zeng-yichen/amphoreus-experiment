/**
 * API client for the Amphoreus backend.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

  submitClientFeedback: (company: string, text: string, postId?: string) =>
    apiFetch<{ status: string; file: string }>("/api/ghostwriter/client-feedback", {
      method: "POST",
      body: JSON.stringify({ company, text, post_id: postId }),
    }),

  submitEditFeedback: (company: string, original: string, revised: string) =>
    apiFetch<{ status: string }>("/api/ghostwriter/feedback", {
      method: "POST",
      body: JSON.stringify({ company, original, revised }),
    }),

  getFeedback: (company: string) =>
    apiFetch<{
      feedback_files: any[];
      feedback_count: number;
      directives: any[];
      directives_count: number;
    }>(`/api/ghostwriter/feedback/${company}`),

  getRunEvents: (runId: string) =>
    apiFetch<{ run: any; events: any[] }>(`/api/ghostwriter/runs/${runId}/events`),

  rollback: (company: string, runId: string) =>
    apiFetch(`/api/ghostwriter/${company}/rollback/${runId}`, { method: "POST" }),

  getFiles: (company: string, path = "") =>
    apiFetch<{ files: any[] }>(`/api/ghostwriter/sandbox/${company}/files?path=${path}`),

  inlineEdit: (company: string, postText: string, instruction: string) =>
    apiFetch<{ result: string }>("/api/ghostwriter/inline-edit", {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, instruction }),
    }),

  getLinkedInUsername: (company: string) =>
    apiFetch<{ username: string | null }>(`/api/ghostwriter/${company}/linkedin-username`),

  saveLinkedInUsername: (company: string, username: string) =>
    apiFetch<{ status: string; username: string }>(`/api/ghostwriter/${company}/linkedin-username`, {
      method: "POST",
      body: JSON.stringify({ username }),
    }),

  getOrdinalUsers: (company: string) =>
    apiFetch<{ users: any[] }>(`/api/ghostwriter/${company}/ordinal-users`),
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

// --- Analyst Findings ---

export const analystApi = {
  getFindings: (company: string) =>
    apiFetch<{ company: string; findings: any[]; last_run: any; age_days: number; total_runs: number }>(
      `/api/strategy/findings/${company}`
    ),
  refresh: (company: string) =>
    apiFetch<{ company: string; findings: any[]; last_run: any }>(
      `/api/strategy/findings/${company}/refresh`
    ),
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

  getHtml: (company: string) =>
    apiFetch<{ html: string | null }>(`/api/strategy/${company}/html`),
};

// --- Progress Report ---

export const reportApi = {
  getData: (company: string, weeks = 2) =>
    apiFetch<any>(`/api/report/${company}?weeks=${weeks}`),

  getHtml: (company: string, weeks = 2) =>
    apiFetch<{ html: string }>(`/api/report/${company}/html?weeks=${weeks}`),
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

  update: (
    postId: string,
    company: string,
    fields: {
      content?: string;
      status?: string;
      title?: string;
      linked_image_id?: string | null;
    }
  ) =>
    apiFetch(`/api/posts/${postId}`, {
      method: "PATCH",
      body: JSON.stringify({ company, ...fields }),
    }),

  delete: (postId: string) =>
    apiFetch(`/api/posts/${postId}`, { method: "DELETE" }),

  rewrite: (postId: string, company: string, postText: string, styleInstruction?: string) =>
    apiFetch<{ result: any }>(`/api/posts/${postId}/rewrite`, {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, style_instruction: styleInstruction }),
    }),

  factCheck: (postId: string, company: string, postText: string) =>
    apiFetch<{
      report: string;
      corrected_post?: string;
      annotated_post?: string;
      citation_comments?: string[];
    }>(`/api/posts/${postId}/fact-check`, {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText }),
    }),

  push: (
    company: string,
    content: string,
    citationComments: string[] = [],
    options?: {
      postId?: string;
      publishAt?: string;
      approvals?: { userId: string; message?: string; dueDate?: string; isBlocking?: boolean }[];
    }
  ) => {
    const base = {
      company,
      ...(options?.publishAt ? { publish_at: options.publishAt } : {}),
      approvals: options?.approvals ?? [],
    };
    const body =
      options?.postId != null && options.postId !== ""
        ? { ...base, post_id: options.postId, content: "" }
        : { ...base, content, citation_comments: citationComments };
    return apiFetch<{ success: boolean; result: string; ordinal_post_ids?: string[] }>(
      "/api/posts/push",
      {
        method: "POST",
        body: JSON.stringify(body),
      }
    );
  },

  pushAll: (
    company: string,
    postsPerMonth: 8 | 12,
    options?: {
      approvals?: { userId: string; message?: string; dueDate?: string; isBlocking?: boolean }[];
    }
  ) =>
    apiFetch<{
      success: boolean;
      pushed: number;
      total: number;
      failed: number;
      cadence: string;
      first_url: string | null;
      errors: string[];
    }>("/api/posts/push-all", {
      method: "POST",
      body: JSON.stringify({
        company,
        posts_per_month: postsPerMonth,
        approvals: options?.approvals ?? [],
      }),
    }),
};

// --- Images ---

export const imagesApi = {
  generate: (
    company: string,
    postText: string,
    model?: string,
    options?: {
      feedback?: string;
      referenceImageId?: string;
      localPostId?: string;
    }
  ) =>
    apiFetch<{ job_id: string; status: string }>("/api/images/generate", {
      method: "POST",
      body: JSON.stringify({
        company,
        post_text: postText,
        model: model ?? "claude-opus-4-6",
        feedback: options?.feedback ?? "",
        reference_image_id: options?.referenceImageId ?? "",
        local_post_id: options?.localPostId ?? "",
      }),
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

// --- Clients ---

export const clientsApi = {
  list: () => apiFetch<{ clients: { slug: string }[] }>("/api/clients"),
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

export const learningApi = {
  listClients: () => apiFetch<{ clients: any[] }>("/api/learning/clients"),
  getClient: (company: string) => apiFetch<any>(`/api/learning/clients/${company}`),
  getCrossClient: () => apiFetch<any>("/api/learning/cross-client"),
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
