/**
 * Backend proxy helpers for route handlers that need to bypass Next.js's
 * built-in ``rewrites()`` layer.
 *
 * Why this exists:
 *   ``next.config.ts`` declares ``source: "/api/:path*" → BACKEND_URL``
 *   which is great for plain JSON calls but uses a buffering proxy that
 *   eats SSE streams (the entire response body is buffered before any
 *   bytes reach the client). For endpoints that stream — ``/sandbox/stream``
 *   specifically — we need a route handler that opens its own fetch to
 *   the backend and pipes the body directly.
 *
 * Route handlers added in ``src/app/api/**`` take precedence over
 * ``rewrites()`` for their exact paths, so adding a handler here does NOT
 * disturb any other proxied endpoint. Every other ``/api/*`` request
 * continues to flow through the default rewrite.
 */

/**
 * Resolve the backend URL. In production, baked as a Docker build ARG
 * that flows through ``next.config.ts`` rewrites. Route handlers can't
 * read that build-time baked value, so we mirror the same env-var
 * contract here (BACKEND_URL), with the same local-dev default.
 */
export function backendUrl(): string {
	const url = process.env.BACKEND_URL || "http://localhost:8000";
	return url.replace(/\/$/, "");
}

/**
 * Headers that should be forwarded from the incoming request to the
 * Python backend. We explicitly allow a small, known set rather than
 * passing through everything — this prevents leaking browser-side
 * headers (``Origin``, ``Referer``, etc.) into the backend's auth layer
 * and keeps the proxy contract explicit.
 *
 * Must stay in sync with ``backend/src/auth/middleware.py`` which
 * looks for: ``cf-access-jwt-assertion``, ``CF_Authorization`` cookie,
 * ``Authorization: Bearer``. We also forward the Lineage-mode headers
 * in case Jacquard's virio-api ever calls through this handler (it
 * currently hits the backend directly; this is defensive).
 */
const FORWARDED_HEADERS = [
	"cf-access-jwt-assertion",
	"authorization",
	"cookie",
	"x-lineage-workspace-url",
	"x-lineage-user-id",
	"x-lineage-user-email",
	"user-agent",
];

/**
 * Extract the subset of request headers that should reach the backend.
 * Case-insensitive; returns a plain object suitable for ``fetch`` headers.
 */
export function forwardedHeaders(req: Request): Record<string, string> {
	const out: Record<string, string> = {};
	for (const name of FORWARDED_HEADERS) {
		const v = req.headers.get(name);
		if (v) out[name] = v;
	}
	return out;
}
