/**
 * Streaming proxy for per-job SSE events (/api/ghostwriter/stream/:job_id).
 *
 * Same rationale and mechanic as
 * ``src/app/api/ghostwriter/sandbox/stream/route.ts`` — see that file for
 * the full justification. Exists because Next.js ``rewrites()`` buffers
 * responses, which breaks SSE.
 */

import { backendUrl, forwardedHeaders } from "@/lib/backend-proxy";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
	req: Request,
	context: { params: Promise<{ job_id: string }> },
): Promise<Response> {
	const { job_id } = await context.params;
	const inUrl = new URL(req.url);
	const target =
		`${backendUrl()}/api/ghostwriter/stream/${encodeURIComponent(job_id)}` +
		(inUrl.search || "");

	let upstream: Response;
	try {
		upstream = await fetch(target, {
			method: "GET",
			headers: {
				...forwardedHeaders(req),
				accept: "text/event-stream",
			},
			signal: req.signal,
			// @ts-expect-error -- undici duplex flag for streaming
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
			{ status: 502, headers: { "content-type": "application/json" } },
		);
	}

	return new Response(upstream.body, {
		status: upstream.status,
		headers: {
			"content-type": "text/event-stream; charset=utf-8",
			"cache-control": "no-cache, no-transform",
			connection: "keep-alive",
			"x-accel-buffering": "no",
		},
	});
}
