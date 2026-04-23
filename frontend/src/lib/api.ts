/**
 * API client for the Amphoreus backend.
 */

// API base URL:
//   - In production (built with NEXT_PUBLIC_API_URL=""), this is an empty
//     string, so fetch("/api/...") goes same-origin → Next.js rewrites proxy
//     to the internal backend (see next.config.ts).
//   - In local dev (var unset), falls back to http://localhost:8000.
// We use ?? instead of || so an explicit empty string is preserved.
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function getAuthHeaders(): Promise<Record<string, string>> {
  // Prefer the live Supabase session — it auto-refreshes the access
  // token as it approaches expiry, so we don't get a cascade of 401s
  // after the default 60-min lifetime runs out. Fall back to the
  // ``access_token`` localStorage key for the narrow windows where the
  // session hasn't hydrated yet (SSR boundary, immediate navigation
  // after /auth/callback).
  let token: string | null = null;
  if (typeof window !== "undefined") {
    try {
      const { getAccessToken } = await import("./supabase");
      token = await getAccessToken();
      if (token) {
        // Keep the localStorage mirror fresh for any code still reading it.
        localStorage.setItem("access_token", token);
      }
    } catch {
      /* supabase module not ready — fall through */
    }
    if (!token) {
      token = localStorage.getItem("access_token");
    }
  }
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    // Critical for Stage 2: forward CF_Authorization cookie so backend can
    // read the Cloudflare Access JWT. Without this, same-origin rewrites skip
    // the cookie and backend 401s every request.
    credentials: "include",
    headers: { ...headers, ...options.headers },
  });
  if (res.status === 401) {
    // Cloudflare Access session expired. Force a full reload — CF will
    // intercept the navigation and re-issue a login PIN challenge.
    if (typeof window !== "undefined") {
      window.location.href = "/";
    }
    throw new Error("Session expired — redirecting to login");
  }
  if (res.status === 403) {
    let detail = "Forbidden";
    try {
      const body = await res.json();
      detail = body.detail || body.error || detail;
    } catch {
      /* ignore */
    }
    throw new Error(`Forbidden: ${detail}`);
  }
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API error ${res.status}: ${error}`);
  }
  return res.json();
}

// --- Shared SSE stream helper ---

async function* streamSSE(url: string, initialAfterId = 0) {
  const headers = await getAuthHeaders();

  // Track the last event ID we saw so we can resume from it on reconnect.
  // The backend's run_events table has auto-incrementing IDs; after a
  // reconnect we only need events newer than what we already yielded.
  // initialAfterId > 0 is how a fresh page mount picks up mid-run events
  // it's already rendered (from the events REST endpoint).
  let lastEventId = initialAfterId;
  let reconnects = 0;
  const MAX_RECONNECTS = 30; // ~5 min of retry at 10s intervals
  let sawTerminal = false;

  while (reconnects <= MAX_RECONNECTS && !sawTerminal) {
    try {
      const connectUrl =
        lastEventId > 0
          ? `${API_BASE}${url}?after_id=${lastEventId}`
          : `${API_BASE}${url}`;
      // credentials:"include" forwards the CF_Authorization cookie on
      // Fly + Cloudflare Access so the reconnect doesn't 401 silently.
      const res = await fetch(connectUrl, { headers, credentials: "include" });
      if (!res.ok || !res.body) {
        // If the job is already done, the endpoint might 404 — stop.
        if (res.status === 404) return;
        reconnects++;
        await new Promise((r) => setTimeout(r, 10_000));
        continue;
      }

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
          // Skip SSE comments (keepalives start with ":")
          if (part.startsWith(":")) continue;
          const line = part.replace(/^data: /, "").trim();
          if (!line) continue;
          try {
            const parsed = JSON.parse(line);
            // Track event ID if present for resume
            if (parsed._event_id && parsed._event_id > lastEventId) {
              lastEventId = parsed._event_id;
            }
            yield parsed;
            // Terminal events end the stream
            if (parsed.type === "done" || parsed.type === "error") {
              sawTerminal = true;
            }
          } catch {
            /* skip malformed */
          }
        }
      }

      // Stream ended (reader returned done). If we already got a terminal
      // event, we're finished. Otherwise the connection dropped mid-stream
      // — reconnect after a brief pause.
      if (sawTerminal) return;
      reconnects++;
      await new Promise((r) => setTimeout(r, 3_000));
    } catch {
      // Network error — retry
      reconnects++;
      await new Promise((r) => setTimeout(r, 10_000));
    }
  }
}

// --- Ghostwriter ---

export const ghostwriterApi = {
  generate: (
    company: string,
    prompt?: string,
    model?: string,
  ) =>
    apiFetch<{ job_id: string; status: string }>("/api/ghostwriter/generate", {
      method: "POST",
      body: JSON.stringify({ company, prompt, model }),
    }),

  streamJob: (jobId: string, afterId = 0) =>
    streamSSE(`/api/ghostwriter/stream/${jobId}`, afterId),

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

  inlineEdit: (company: string, postText: string, instruction: string) =>
    apiFetch<{ result: string }>("/api/ghostwriter/inline-edit", {
      method: "POST",
      body: JSON.stringify({ company, post_text: postText, instruction }),
    }),

  /** Read-only. Resolved server-side from Jacquard's
   *  ``users.linkedin_url``, with a legacy file fallback for slugs
   *  that predate the per-FOC dropdown. Writing the handle from the
   *  app was retired — edit in Jacquard instead. */
  getLinkedInUsername: (company: string) =>
    apiFetch<{ username: string | null }>(`/api/ghostwriter/${company}/linkedin-username`),

  getOrdinalUsers: (company: string) =>
    apiFetch<{ users: any[] }>(`/api/ghostwriter/${company}/ordinal-users`),

  // Calendar
  getCalendar: (company: string, month?: string) =>
    apiFetch<{ company: string; month: string | null; posts: any[] }>(
      `/api/ghostwriter/${company}/calendar${month ? `?month=${month}` : ""}`
    ),

  schedulePost: (company: string, postId: string, scheduledDate: string | null) =>
    apiFetch<any>(`/api/ghostwriter/${company}/posts/${postId}/schedule`, {
      method: "PATCH",
      body: JSON.stringify({ scheduled_date: scheduledDate }),
    }),

  autoAssign: (company: string, cadence: string, startDate?: string) =>
    apiFetch<{ assigned: number; posts: any[] }>(
      `/api/ghostwriter/${company}/calendar/auto-assign`,
      {
        method: "POST",
        body: JSON.stringify({ cadence, start_date: startDate }),
      }
    ),

  /**
   * @deprecated 2026-04-23 — Ordinal outbound disabled. Backend
   * returns HTTP 410 Gone. Virio is churning off Ordinal; the
   * replacement publishing pipeline will land behind a new endpoint.
   * Calendar scheduling (creating/editing ``scheduled_date`` on
   * drafts) still works — only the "push to Ordinal" step is gone.
   */
  pushAll: (company: string) =>
    apiFetch<{ pushed: number; results: any[] }>(
      `/api/ghostwriter/${company}/calendar/push-all`,
      { method: "POST" }
    ),

  /**
   * @deprecated 2026-04-23 — Ordinal outbound disabled. See {@link pushAll}.
   */
  pushSingle: (company: string, postId: string) =>
    apiFetch<{ id: string; status: string; ordinal_post_id?: string }>(
      `/api/ghostwriter/${company}/calendar/push-single`,
      {
        method: "POST",
        body: JSON.stringify({ post_id: postId }),
      }
    ),
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

// --- Cyrene (Strategic Growth Agent) ---

export const cyreneApi = {
  run: (company: string) =>
    apiFetch<{ job_id: string }>(`/api/strategy/cyrene/${company}`, {
      method: "POST",
    }),

  streamJob: (jobId: string, afterId = 0) =>
    streamSSE(`/api/strategy/stream/${jobId}`, afterId),

  getBrief: (company: string) =>
    apiFetch<any>(`/api/strategy/cyrene/${company}/brief`),
};

// --- Progress Report ---

export const reportApi = {
  getData: (company: string, weeks = 2) =>
    apiFetch<any>(`/api/report/${company}?weeks=${weeks}`),

  getHtml: (company: string, weeks = 2) =>
    apiFetch<{ html: string }>(`/api/report/${company}/html?weeks=${weeks}`),

  generate: (company: string, weeks = 2) =>
    apiFetch<{ job_id: string; status: string }>(
      `/api/report/${company}/generate?weeks=${weeks}`,
      { method: "POST" },
    ),

  streamJob: (jobId: string, afterId = 0) =>
    streamSSE(`/api/report/stream/${jobId}`, afterId),

  renderedUrl: (company: string) => `${API_BASE}/api/report/${company}/rendered`,
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

  // Reject — state transition (NOT delete). Preserves the local_posts
  // row and every paired draft_feedback row so the draft surfaces in
  // Stelle's / Aglaea's post bundle under the "Rejected" class with
  // the comments that caused the rejection paired line-by-line.
  // Use after manually deleting the draft from Ordinal.
  reject: (postId: string, reason?: string) =>
    apiFetch<{ rejected: true; post_id: string }>(`/api/posts/${postId}/reject`, {
      method: "POST",
      body: JSON.stringify({ reason: reason ?? null }),
    }),

  // Manually pair a Stelle draft to its published LinkedIn post by
  // date. ``publishDate`` is a YYYY-MM-DD string interpreted in
  // America/Los_Angeles (PST/PDT). The server finds the single
  // linkedin_posts row from this FOC on that LA calendar day and
  // stamps ``matched_provider_urn`` on the draft — which then lets
  // post_bundle render DELTA (draft → published). Returns 409 when
  // zero matches (not scraped yet) or multiple matches (posted
  // twice same day) so the caller can surface a useful message.
  setPublishDate: (postId: string, publishDate: string) =>
    apiFetch<{
      paired: true;
      post_id: string;
      matched_provider_urn: string;
      matched_posted_at: string;
      matched_hook: string;
      matched_reactions: number;
    }>(`/api/posts/${postId}/set-publish-date`, {
      method: "POST",
      body: JSON.stringify({ publish_date: publishDate }),
    }),

  unsetPublishDate: (postId: string) =>
    apiFetch<{ unpaired: true; post_id: string }>(
      `/api/posts/${postId}/set-publish-date`,
      { method: "DELETE" },
    ),

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

  /**
   * @deprecated 2026-04-23 — Ordinal outbound disabled. Backend now
   * returns HTTP 410 Gone. Virio is churning off Ordinal; drafts stay
   * in local_posts. UI should hide all "Push to Ordinal" affordances.
   */
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

  /**
   * @deprecated 2026-04-23 — Ordinal outbound disabled. Backend now
   * returns HTTP 410 Gone. See {@link push} for rationale.
   */
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

  // ---- Feedback (draft_feedback) ----
  // Content-engineer comments on a draft. Post-wide = null selection;
  // inline = start/end/text from the browser's window.getSelection().
  // Unresolved rows are injected into Cyrene.rewrite_single_post so the
  // next "Rewrite" actually addresses the notes.
  listComments: (postId: string, includeResolved = true) =>
    apiFetch<{ comments: Comment[] }>(
      `/api/posts/${postId}/comments?include_resolved=${includeResolved}`,
    ),

  addComment: (
    postId: string,
    body: string,
    options?: {
      source?: "operator_postwide" | "operator_inline";
      authorEmail?: string | null;
      authorName?: string | null;
      selection?: { start: number; end: number; text: string };
    },
  ) =>
    apiFetch<{ comment: Comment }>(`/api/posts/${postId}/comments`, {
      method: "POST",
      body: JSON.stringify({
        body,
        source: options?.selection ? "operator_inline" : (options?.source ?? "operator_postwide"),
        author_email: options?.authorEmail ?? null,
        author_name: options?.authorName ?? null,
        selection_start: options?.selection?.start ?? null,
        selection_end: options?.selection?.end ?? null,
        selected_text: options?.selection?.text ?? null,
      }),
    }),

  editComment: (postId: string, commentId: string, body: string) =>
    apiFetch<{ updated: true; id: string }>(
      `/api/posts/${postId}/comments/${commentId}`,
      { method: "PATCH", body: JSON.stringify({ body }) },
    ),

  resolveComment: (postId: string, commentId: string, resolvedBy?: string) =>
    apiFetch<{ resolved: true; id: string }>(
      `/api/posts/${postId}/comments/${commentId}/resolve${resolvedBy ? `?resolved_by=${encodeURIComponent(resolvedBy)}` : ""}`,
      { method: "POST" },
    ),

  deleteComment: (postId: string, commentId: string) =>
    apiFetch<{ deleted: true }>(
      `/api/posts/${postId}/comments/${commentId}`,
      { method: "DELETE" },
    ),

  revertToOriginal: (postId: string) =>
    apiFetch<{ reverted: true; post: any }>(`/api/posts/${postId}/revert`, {
      method: "POST",
    }),

  listRevisions: (postId: string, limit = 50) =>
    apiFetch<{ revisions: Revision[] }>(
      `/api/posts/${postId}/revisions?limit=${limit}`,
    ),
};

export type Comment = {
  id: string;
  draft_id: string;
  source: "operator_postwide" | "operator_inline" | "ordinal" | "slack" | "client_email";
  author_email: string | null;
  author_name: string | null;
  body: string;
  selection_start: number | null;
  selection_end: number | null;
  selected_text: string | null;
  resolved: boolean;
  resolved_at: string | null;
  resolved_by: string | null;
  created_at: string;
};

export type Revision = {
  id: string;
  draft_id: string;
  source: "stelle_initial" | "castorice_rewrite" | "operator_edit" | "revert_to_original" | "rewrite_with_feedback";
  author_email: string | null;
  created_at: string;
  content: string;
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

// --- Clients ---

export const clientsApi = {
  list: () => apiFetch<{ clients: { slug: string }[] }>("/api/clients"),
};

// --- Me (Stage 2 auth) ---

export type MeResponse = {
  email: string;
  is_admin: boolean;
  allowed_clients: "*" | string[];
  auth_enabled: boolean;
};

export const meApi = {
  get: () => apiFetch<MeResponse>("/api/me"),
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

// --- Transcripts (add-only, backed by the Amphoreus mirror) ---
//
// The UI talks exclusively to /api/mirror/transcripts. Jacquard-synced
// rows and Amphoreus-uploaded rows come back through the same list
// endpoint with a ``source`` discriminator. Add paths (upload + paste)
// always write to the Amphoreus mirror — never to fly-local, never to
// Jacquard.

/** Row shape returned by GET /api/mirror/transcripts. */
export type MirrorTranscript = {
  id: string;
  name: string;
  source: "amphoreus" | "jacquard";
  created_at: string | null;        // ISO
  start_time: string | null;        // ISO, from Jacquard rows when present
  duration_seconds: number | null;
  description: string | null;       // Amphoreus rows only
  filename: string | null;          // Amphoreus rows only
  uploaded_by: string | null;       // Amphoreus rows only
  mount: "transcripts" | "context"; // which Stelle mount surfaces this row
  /** For context-mount rows only: the FOC this context was uploaded
   *  for. ``null`` means shared across every FOC at the company
   *  (Jacquard-mirrored rows + legacy Amphoreus uploads made before
   *  per-FOC scoping). Non-null means the row is visible only to that
   *  specific FOC's agents. Used by the UI to render a shared/for-foc
   *  badge so operators can tell at a glance whether an uploaded doc
   *  leaks across teammates at shared-slug companies like Trimble. */
  user_id?: string | null;
  /** Finer-grained meeting tag for Tribbie-local uploads.
   *  content_interview — direct client interview (high-signal)
   *  sync               — ops check-in / weekly GTM call (lower-signal)
   *  other              — anything else tagged explicitly
   *  null               — legacy rows, Jacquard-sourced rows, context uploads */
  meeting_subtype: "content_interview" | "sync" | "other" | null;
};

export type MeetingSubtype = "content_interview" | "sync" | "other";

/**
 * @deprecated Legacy fly-local shape. Kept only for any caller that still
 * imports it; new code uses {@link MirrorTranscript}.
 */
export type TranscriptFile = {
  filename: string;
  size_bytes: number;
  modified_at: number;
  source_label: string | null;
  uploaded_by: string | null;
  uploaded_at: number | null;
  original_filename: string | null;
  content_type: string | null;
};

/** Routing-kind for Amphoreus uploads.
 *  - ``transcript`` / ``call``: dialogue content → ``meetings`` → Stelle sees
 *    it at ``<slug>/transcripts/``.
 *  - ``article`` / ``briefing`` / ``notes``: reference content → ``context_files``
 *    → Stelle sees it at ``<slug>/context/``. */
export type UploadKind = "transcript" | "call" | "article" | "briefing" | "notes";

/** Response shape from both ``uploadToMirror`` and ``pasteToMirror``. */
export type MirrorUploadResponse = {
  id: string;
  company_id: string;
  storage_path: string;
  size_bytes: number;
  filename: string;
  description: string;
  source: "amphoreus";
  /** Present when the upload routed to ``context_files``. */
  kind?: UploadKind;
  /** Present when the upload routed to ``context_files``. */
  target?: "context_files";
};

export const transcriptsApi = {
  /**
   * List every transcript the Amphoreus mirror knows about for this client.
   * Returns Jacquard-synced + Amphoreus-uploaded rows merged, deduped on PK,
   * newest first. Each row carries a ``source`` field so the UI can badge.
   */
  list: (company: string, limit = 100) =>
    apiFetch<{ company_id: string; transcripts: MirrorTranscript[] }>(
      `/api/mirror/transcripts?company=${encodeURIComponent(company)}&limit=${limit}`,
    ),

  /**
   * Upload a file into the Amphoreus mirror.
   * Body → Supabase Storage. Row → ``meetings`` with ``_source='amphoreus'``.
   * Never writes back to Jacquard.
   */
  uploadToMirror: async (
    company: string,
    file: File,
    description: string,
    userId?: string,
    kind: UploadKind = "transcript",
    meetingSubtype?: MeetingSubtype,
  ) => {
    const form = new FormData();
    form.append("company", company);
    form.append("description", description);
    form.append("kind", kind);
    if (userId) form.append("user_id", userId);
    if (meetingSubtype) form.append("meeting_subtype", meetingSubtype);
    form.append("file", file);
    // CRITICAL: strip Content-Type so fetch auto-generates the
    // ``multipart/form-data; boundary=...`` header from the FormData
    // body. getAuthHeaders() defaults Content-Type to application/json,
    // which (when not stripped) overrides the boundary and causes
    // FastAPI to see an empty body — 422 "field required" on every
    // Form/File param. Keep Authorization + any other auth headers;
    // drop only Content-Type.
    const authHeaders = await getAuthHeaders();
    const { "Content-Type": _drop, ...rest } = authHeaders;
    void _drop;
    const res = await fetch(`${API_BASE}/api/mirror/transcripts`, {
      method: "POST",
      credentials: "include",
      headers: rest,
      body: form,
    });
    if (res.status === 401) {
      if (typeof window !== "undefined") window.location.href = "/";
      throw new Error("Session expired");
    }
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Mirror upload failed (${res.status}): ${detail}`);
    }
    return res.json() as Promise<MirrorUploadResponse>;
  },

  /**
   * Paste raw text as a new Amphoreus-sourced upload. ``kind`` routes
   * transcripts to ``meetings`` and articles/briefings/notes to
   * ``context_files``. Default is ``transcript`` for back-compat.
   */
  pasteToMirror: (
    company: string,
    text: string,
    description: string,
    userId?: string,
    kind: UploadKind = "transcript",
    meetingSubtype?: MeetingSubtype,
  ) =>
    apiFetch<MirrorUploadResponse>(`/api/mirror/transcripts/paste`, {
      method: "POST",
      body: JSON.stringify({
        company,
        text,
        description,
        kind,
        ...(userId ? { user_id: userId } : {}),
        ...(meetingSubtype ? { meeting_subtype: meetingSubtype } : {}),
      }),
    }),

  /**
   * Delete an Amphoreus-uploaded transcript by id. Admin-only; the
   * backend refuses to delete Jacquard-sourced rows (they'd resync).
   * Wired to the trash-can icon on the transcripts page for cases
   * where a content engineer uploaded to the wrong client.
   */
  deleteFromMirror: (transcriptId: string) =>
    apiFetch<{ deleted: true; id: string; storage_removed: boolean }>(
      `/api/mirror/transcripts/${encodeURIComponent(transcriptId)}`,
      { method: "DELETE" },
    ),

  /**
   * Upload an image pasted into a textarea. Backend stores the blob
   * under context_files + generates an AI description so agents can
   * "see" the image via extracted_text. Returns a ``marker_text``
   * markdown reference the caller inserts into the textarea at the
   * cursor so the surrounding transcript text stays linked to the
   * image. Image rows are stamped with the same ``user_id`` as a
   * regular transcript upload (resolved from pseudo-slug).
   */
  pasteImageToMirror: (
    company: string,
    imageBase64: string,
    contentType: string,
    opts?: { filename?: string; description?: string; userId?: string },
  ) =>
    apiFetch<{
      id: string;
      company_id: string;
      storage_path: string;
      filename: string;
      description: string;
      marker_text: string;
      size_bytes: number;
      target: "context_files";
      source: "amphoreus";
    }>(`/api/mirror/transcripts/paste-image`, {
      method: "POST",
      body: JSON.stringify({
        company,
        content_type: contentType,
        image_base64: imageBase64,
        ...(opts?.filename ? { filename: opts.filename } : {}),
        ...(opts?.description ? { description: opts.description } : {}),
        ...(opts?.userId ? { user_id: opts.userId } : {}),
      }),
    }),

  /**
   * @deprecated Fly-local paste. Use {@link pasteToMirror}.
   */
  paste: (company: string, text: string, sourceLabel: string) =>
    apiFetch<{
      status: string;
      filename: string;
      size_bytes: number;
      source_label: string;
    }>(`/api/transcripts/${company}/paste`, {
      method: "POST",
      body: JSON.stringify({ text, source_label: sourceLabel }),
    }),

  /**
   * @deprecated Fly-local delete; localhost-only on the backend. UI no longer
   * surfaces a delete button. Clean up Amphoreus-sourced rows via the
   * Supabase dashboard until a mirror-delete endpoint lands.
   */
  delete: (company: string, filename: string) =>
    apiFetch<{ status: string; filename: string }>(
      `/api/transcripts/${company}/${encodeURIComponent(filename)}`,
      { method: "DELETE" },
    ),
};

// --- Usage / Spend (admin-only) ---

export type UsageUserRow = {
  user_email: string | null;
  n_calls: number;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cost_usd: number;
};

export type UsageClientRow = {
  client_slug: string | null;
  n_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

export type UsageModelRow = {
  model: string;
  provider: string;
  n_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

export type UsageSummary = {
  since: number;
  until: number;
  total: {
    n_calls: number;
    input_tokens: number;
    output_tokens: number;
    cost_usd: number;
  };
  by_user: UsageUserRow[];
};

function usageQueryString(opts: { since?: string; until?: string }): string {
  const params = new URLSearchParams();
  if (opts.since) params.set("since", opts.since);
  if (opts.until) params.set("until", opts.until);
  const s = params.toString();
  return s ? `?${s}` : "";
}

export const usageApi = {
  summary: (opts: { since?: string; until?: string } = {}) =>
    apiFetch<UsageSummary>(`/api/usage/summary${usageQueryString(opts)}`),

  byClient: (opts: { since?: string; until?: string } = {}) =>
    apiFetch<{ since: number; until: number; by_client: UsageClientRow[] }>(
      `/api/usage/by-client${usageQueryString(opts)}`
    ),

  byUser: (email: string, opts: { since?: string; until?: string } = {}) =>
    apiFetch<{
      user_email: string | null;
      since: number;
      until: number;
      by_model: UsageModelRow[];
    }>(`/api/usage/by-user/${encodeURIComponent(email)}${usageQueryString(opts)}`),
};

// --- Deploy (localhost only) ---

export const deployApi = {
  code: (target: string = "both") =>
    apiFetch<{ status: string; target: string; key: string }>("/api/deploy/code", {
      method: "POST",
      body: JSON.stringify({ target }),
    }),

  status: (key: string) =>
    apiFetch<{ key: string; status: string; log: string; returncode?: number }>(
      `/api/deploy/status/${encodeURIComponent(key)}`
    ),
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
