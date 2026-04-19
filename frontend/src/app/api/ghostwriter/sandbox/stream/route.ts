/**
 * Streaming proxy for Stelle's SSE sandbox events.
 *
 * Why this route handler exists:
 *   Next.js's built-in ``rewrites()`` forwards responses through an
 *   internal buffering proxy that doesn't flush Server-Sent Events in
 *   real-time — the whole response body is buffered before any bytes
 *   reach the client. For ``/sandbox/stream`` that means the Lineage
 *   terminal (which consumes this endpoint via virio-api's proxyStream)
 *   never sees any tool-call events even while Stelle runs correctly
 *   on Fly.
 *
 * What this does instead:
 *   - Opens its own fetch to ``${BACKEND_URL}/api/ghostwriter/sandbox/stream``
 *     on the Python backend.
 *   - Returns the upstream ``ReadableStream`` directly in a new Response
 *     so bytes flow through as the backend emits them.
 *   - Forwards the handful of auth headers the backend actually looks at
 *     (CF Access JWT, Authorization: Bearer, session cookie, Lineage
 *     headers for completeness). No full passthrough — keeps the contract
 *     explicit.
 *
 * Every OTHER /api/* path continues to use the default ``rewrites()``
 * proxy from ``next.config.ts`` — plain JSON calls don't need special
 * handling, and we want to keep the surface area small.
 */

import { backendUrl, forwardedHeaders } from "@/lib/backend-proxy";

// Route handlers default to dynamic rendering in Next.js 15, but being
// explicit here prevents any future static-optimization pass from
// caching the handler or its response body.
export const dynamic = "force-dynamic";

// nodejs runtime (not edge) — we need to pipe a long-lived stream and
// edge runtime has execution-time limits that would truncate it.
export const runtime = "nodejs";

export async function GET(req: Request): Promise<Response> {
	const inUrl = new URL(req.url);
	const target = `${backendUrl()}/api/ghostwriter/sandbox/stream?${inUrl.searchParams}`;

	let upstream: Response;
	try {
		upstream = await fetch(target, {
			method: "GET",
			headers: {
				...forwardedHeaders(req),
				accept: "text/event-stream",
			},
			// Disable fetch's automatic decompression/buffering behavior
			// where possible. Node's undici implementation streams the body
			// directly when we don't read it eagerly — the key is passing
			// ``upstream.body`` through without touching it.
			signal: req.signal,
			// @ts-expect-error -- undici-specific but widely supported
			duplex: "half",
		});
	} catch (err) {
		return new Response(
			JSON.stringify({
				error: "upstream unreachable",
				detail: err instanceof Error ? err.message : String(err),
			}),
			{
				status: 502,
				headers: { "content-type": "application/json" },
			},
		);
	}

	if (!upstream.body) {
		return new Response(
			JSON.stringify({ error: "upstream returned no body" }),
			{
				status: 502,
				headers: { "content-type": "application/json" },
			},
		);
	}

	return new Response(upstream.body, {
		status: upstream.status,
		headers: {
			"content-type": "text/event-stream; charset=utf-8",
			"cache-control": "no-cache, no-transform",
			connection: "keep-alive",
			// Disable Nginx / Cloudflare buffering of the response body.
			// Redundant with ``cache-control: no-transform`` but defensive.
			"x-accel-buffering": "no",
		},
	});
}
