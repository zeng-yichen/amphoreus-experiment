# Amphoreus Deploy Runbook — Stage 1

End state: 4 named engineers can reach a hosted Amphoreus UI at
`https://cyrene.virio.ai`. Nobody else can. Code is not exposed.
Backend is not publicly reachable.

## Architecture

```
Engineer's browser
      │
      ▼
Cloudflare Access (email allowlist, 4 emails)
      │
      ▼
cyrene.fly.dev  (Next.js, public on 443)
      │  Next.js rewrites /api/*
      ▼
amphoreus.internal:8000  (FastAPI, Fly 6PN only)
      │
      ▼
/data  (persistent Fly volume)
  ├─ memory/        (per-client state, embeddings, directives)
  ├─ sqlite/        (runs, events, local_posts, ruan_mei_state)
  ├─ backend_data/  (engager history)
  ├─ images/        (static image serving)
  ├─ products/      (test data)
  └─ output/        (generated reports)
```

**No code changes to the Python app were made.** All path wiring is done
via `backend/docker-entrypoint.sh`, which symlinks `/data/*` into the
in-image paths at container start.

---

## Prerequisites

- Fly.io account + `flyctl` installed (`brew install flyctl`)
- Cloudflare account + a domain you control (for Access)
- All the API keys currently in local `.env`
- This repo cloned, current branch clean

Log in once:

```sh
fly auth login
```

---

## 1. Create the backend app + volume

From the **repo root**:

```sh
# Create the app (no deploy yet)
fly apps create amphoreus --org personal   # or your org slug

# Create the persistent volume (10 GB is plenty for Stage 1)
fly volumes create amphoreus_data \
    --app    amphoreus \
    --region iad \
    --size   10
```

Set every secret the backend needs. Do NOT commit these anywhere. Copy
the values from your local `.env`:

```sh
fly secrets set --app amphoreus \
    ANTHROPIC_API_KEY="sk-ant-..." \
    OPENAI_API_KEY="sk-proj-..." \
    GEMINI_API_KEY="AIza..." \
    SUPABASE_URL="https://xxxx.supabase.co" \
    SUPABASE_KEY="eyJ..." \
    APIFY_API_TOKEN="apify_api_..." \
    SERPER_API_KEY="..." \
    SERPER_BASE_URL="https://google.serper.dev/search" \
    PERPLEXITY_API_KEY="pplx-..." \
    PERPLEXITY_BASE_URL="https://api.perplexity.ai" \
    ORDINAL_AGENCY_KEY="ord_c_..." \
    ORDINAL_API_KEY="ord_c_..." \
    PARALLEL_API_KEY="q-..." \
    PINECONE_API_KEY="pcsk_..." \
    PINECONE_INDEX="user-posts" \
    ELEVENLABS_API_KEY="sk_..." \
    JWT_SECRET="$(openssl rand -hex 32)"
```

> **Note:** `ORDINAL_AGENCY_KEY` and `ORDINAL_API_KEY` are set to the same
> value because the codebase currently reads both names in different
> places. Cleanup is a Stage 2 concern.

Deploy:

```sh
fly deploy --config backend/fly.toml --app amphoreus
```

Expected: one machine in `iad`, volume mounted at `/data`, `/health`
returning 200 on the internal address. The app has NO public HTTP handler,
so `curl https://amphoreus.fly.dev` will fail — that's correct.

Verify from inside the Fly network:

```sh
fly ssh console --app amphoreus
# inside the machine:
curl -s http://127.0.0.1:8000/health
# → {"status":"ok","service":"amphoreus"}
ls -la /data
# → memory/ sqlite/ backend_data/ images/ products/ output/
exit
```

---

## 2. Migrate existing `memory/` and SQLite onto the volume

**This is the step that makes the engineers see real client data.** Until
you do this, the hosted instance is empty.

From your laptop, with Fly SSH:

```sh
# Copy local memory/ to the volume (this may take a few minutes)
fly ssh sftp shell --app amphoreus
# inside the sftp shell:
put -r /Users/zengyichen/virio/amphoreus-experiment/memory /data
put -r /Users/zengyichen/virio/amphoreus-experiment/backend/data /data/backend_data_local
put -r /Users/zengyichen/virio/amphoreus-experiment/data /data/sqlite_local
bye
```

Then rearrange on the machine so the paths match what the entrypoint
expects:

```sh
fly ssh console --app amphoreus
cd /data

# Merge the uploaded dirs into the expected layout
# (memory/ is already correct)
mv /data/backend_data_local/* /data/backend_data/ 2>/dev/null || true
mv /data/sqlite_local/amphoreus.db /data/sqlite/amphoreus.db 2>/dev/null || true
rmdir /data/backend_data_local /data/sqlite_local 2>/dev/null || true

ls -la /data/memory | head
ls -la /data/sqlite
exit
```

Restart the backend to pick up the migrated state:

```sh
fly machine restart --app amphoreus
```

Watch logs to confirm clean startup and SQLite init:

```sh
fly logs --app amphoreus
# Look for:
#   [entrypoint] SQLITE_PATH=/data/sqlite/amphoreus.db
#   SQLite initialized at /data/sqlite/amphoreus.db
#   Ordinal sync loop started.
```

---

## 3. Create the frontend app

From the **`frontend/` dir**:

```sh
cd frontend
fly apps create cyrene --org personal
```

Secrets the frontend needs at runtime (for Supabase client-side auth if
you still use it — otherwise skip):

```sh
fly secrets set --app cyrene \
    NEXT_PUBLIC_SUPABASE_URL="https://xxxx.supabase.co" \
    NEXT_PUBLIC_SUPABASE_ANON_KEY="eyJ..."
```

Deploy:

```sh
fly deploy --config fly.toml --app cyrene
```

Test the raw Fly URL (before Cloudflare Access is in front):

```sh
curl -I https://cyrene.fly.dev/
# → HTTP/2 200
```

Test that the proxy to the backend works. `/health` is at the root of the
backend (no `/api` prefix), so use a real `/api/*` router to verify the
Next.js rewrite:

```sh
curl -s https://cyrene.fly.dev/api/clients
# → {"clients":[...]}  (empty array if memory/ hasn't been migrated yet)
```

**At this point the app is live on the public internet with no auth.**
Don't share the URL yet. The next step locks it down.

---

## 4. Put Cloudflare Access in front

### 4a. DNS

In Cloudflare, add a CNAME for the domain you want to use:

```
cyrene.virio.ai   CNAME   cyrene.fly.dev   (Proxied — orange cloud)
```

Then tell Fly about the custom domain + issue a cert:

```sh
fly certs add cyrene.virio.ai --app cyrene
fly certs show cyrene.virio.ai --app cyrene
# wait for the cert to show as "Ready"
```

Confirm the domain serves the frontend:

```sh
curl -I https://cyrene.virio.ai/
# → HTTP/2 200
```

### 4b. Enable Cloudflare Access (Zero Trust)

In the Cloudflare dashboard:

1. **Zero Trust → Access → Applications → Add an application → Self-hosted**
2. **Application name:** `Amphoreus`
3. **Session duration:** `24 hours` (engineers re-auth daily)
4. **Application domain:** `cyrene.virio.ai`
5. **Identity providers:** enable **One-time PIN** at minimum. Optionally
   add Google or GitHub.
6. Click **Next → Add policy:**
   - **Policy name:** `Content Engineers`
   - **Action:** `Allow`
   - **Configure rules → Include → Emails**, add the 4 engineer addresses:
     ```
     alice@virio.com
     bob@virio.com
     carol@virio.com
     dan@virio.com
     ```
7. **Next → Add application**

Test: open an incognito window and visit `https://cyrene.virio.ai`.
You should land on a Cloudflare Access login prompt. Sign in with a
whitelisted email → you're in. Try with a non-whitelisted email → denied.

### 4c. (Recommended) Lock the Fly origin to Cloudflare only

Right now, someone who guesses `https://cyrene.fly.dev` can
bypass Cloudflare entirely. Two options:

**Option A — simplest: rely on secrecy.** The Fly URL is long and unguessed.
Acceptable for Stage 1 behind a private beta. Move on.

**Option B — proper lock-down: Cloudflare Tunnel.** Replace the public Fly
service with a `cloudflared` sidecar that opens an outbound tunnel to CF.
This is a Stage 2 task; added to `NEXT_STEPS.md`.

For now, go with Option A.

---

## 5. Share the URL with the 4 engineers

Send them:

```
URL:     https://cyrene.virio.ai
How:     You'll get a Cloudflare Access email-PIN login the first time.
         Enter the PIN from your email. 24-hour session after that.
Scope:   Stage 1 — everyone sees every client. Per-client scoping lands
         in Stage 2.
Support: <your channel>
```

---

## 6. Day-2 operations

### Redeploy after code changes

```sh
# Backend
fly deploy --config backend/fly.toml --app amphoreus

# Frontend
cd frontend && fly deploy --config fly.toml --app cyrene
```

The backend has a 10GB volume that persists across deploys — you're not
going to lose `memory/` or SQLite state.

### Read logs

```sh
fly logs --app amphoreus
fly logs --app cyrene
```

### SSH into the backend to inspect state

```sh
fly ssh console --app amphoreus
ls /data/memory
sqlite3 /data/sqlite/amphoreus.db "SELECT id, client_slug, agent, status FROM runs ORDER BY created_at DESC LIMIT 10;"
exit
```

### Back up the volume

```sh
# Fly volumes support snapshots
fly volumes snapshots list --app amphoreus
fly volumes snapshots create <volume-id> --app amphoreus
```

Schedule weekly snapshots (Stage 2 task).

### Add / remove an engineer

**Don't redeploy.** Just edit the Cloudflare Access policy:

- Zero Trust → Access → Applications → Amphoreus → Policies → Content
  Engineers → Edit → add/remove email → Save.

Takes effect within ~30 seconds.

### Rotate a secret

```sh
fly secrets set --app amphoreus ANTHROPIC_API_KEY="sk-ant-NEW..."
# Fly restarts the machine automatically after secrets change.
```

---

## What Stage 1 deliberately does NOT do

- **No per-client scoping.** All 4 engineers can see all clients. Stage 2
  adds an ACL layer (see `NEXT_STEPS.md`).
- **No audit log of per-user actions.** Cloudflare Access logs sign-ins;
  in-app actions are not attributed to a user yet.
- **No JWT verification on the backend.** The backend trusts that
  traffic came via the frontend (which is only reachable through CF Access).
  Stage 2 adds `Cf-Access-Jwt-Assertion` verification.
- **No Fly origin lockdown.** See §4c above.
- **No CI/CD.** Deploys are manual `fly deploy`. Wire up GitHub Actions
  when the cadence justifies it.

---

## Rollback

If a deploy breaks:

```sh
fly releases --app amphoreus           # find the last good release
fly deploy --image <previous-image-ref> --app amphoreus
```

Or scale to zero to take it offline:

```sh
fly scale count 0 --app cyrene
# app is now unreachable. Scale back up with:
fly scale count 1 --app cyrene
```
