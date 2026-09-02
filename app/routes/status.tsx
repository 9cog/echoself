import { json, type MetaFunction } from "@remix-run/node";
import { Link, useLoaderData, useRevalidator } from "@remix-run/react";
import { getDTEChatEngine } from "~/services/dte-chat.server";

export const meta: MetaFunction = () => [{ title: "Deep Tree Echo - Status" }];

export async function loader() {
  const status = await getDTEChatEngine().getStatus();
  return json({ status, generatedAt: new Date().toISOString() });
}

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card text-card-foreground p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide opacity-70 mb-3">
        {title}
      </h2>
      {children}
    </section>
  );
}

function Dot({ ok, warn }: { ok: boolean; warn?: boolean }) {
  const cls = ok ? "bg-green-500" : warn ? "bg-yellow-500" : "bg-red-500";
  return <span className={`inline-block w-2.5 h-2.5 rounded-full ${cls}`} />;
}

function fmtUptime(ms: number) {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m ${s % 60}s`;
}

export default function StatusPage() {
  const { status, generatedAt } = useLoaderData<typeof loader>();
  const revalidator = useRevalidator();
  const t = status.training;

  return (
    <div className="min-h-screen p-6 max-w-5xl mx-auto">
      <header className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Deep Tree Echo · Status</h1>
          <p className="text-xs opacity-60">
            generated {new Date(generatedAt).toLocaleString()} · engine uptime{" "}
            {fmtUptime(status.uptimeMs)}
          </p>
        </div>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={() => revalidator.revalidate()}
            className="px-3 py-1.5 text-sm border border-border rounded-md hover:bg-primary/10"
          >
            {revalidator.state === "loading" ? "refreshing…" : "refresh"}
          </button>
          <Link
            to="/chat"
            className="px-3 py-1.5 text-sm bg-primary text-white rounded-md"
          >
            open chat
          </Link>
        </div>
      </header>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Language model providers">
          {status.providers.length === 0 ? (
            <p className="text-sm text-yellow-400">
              No providers configured. The chat answers from the local persona
              fallback. Set OLLAMA_BASE_URL, OPENAI_API_KEY or
              ANTHROPIC_API_KEY.
            </p>
          ) : (
            <ul className="space-y-2 text-sm">
              {status.providers.map(p => (
                <li key={p.kind} className="flex items-start gap-2">
                  <Dot
                    ok={p.successes > 0 && !p.lastError}
                    warn={p.successes === 0 && p.failures === 0}
                  />
                  <div className="flex-1">
                    <div className="flex justify-between">
                      <span className="font-medium">
                        {p.kind}{" "}
                        <span className="opacity-60 font-mono text-xs">
                          {p.model}
                        </span>
                      </span>
                      <span className="opacity-70 font-mono text-xs">
                        {p.successes} ok / {p.failures} failed
                      </span>
                    </div>
                    {p.lastError && (
                      <div className="text-xs text-red-400 break-all">
                        {p.lastError}
                      </div>
                    )}
                    {p.lastOk && (
                      <div className="text-xs opacity-60">
                        last ok {new Date(p.lastOk).toLocaleTimeString()}
                      </div>
                    )}
                    {p.successes === 0 && p.failures === 0 && (
                      <div className="text-xs opacity-60">
                        not yet exercised — send a chat message
                      </div>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-3 text-xs opacity-70">
            live provider:{" "}
            <span
              className={
                status.hasLiveProvider ? "text-green-400" : "text-yellow-400"
              }
            >
              {status.hasLiveProvider ? "yes" : "no"}
            </span>
          </p>
        </Card>

        <Card title="Memory">
          <div className="flex items-center gap-2 text-sm mb-2">
            <Dot ok={status.memory.enabled} warn={!status.memory.enabled} />
            <span>
              backend <span className="font-mono">{status.memory.backend}</span>
            </span>
          </div>
          <p className="text-3xl font-bold">{status.memory.count}</p>
          <p className="text-xs opacity-60">dte-chat memories stored</p>
          <p className="text-xs opacity-60 mt-2">
            {status.sessions} active cognitive session(s) in this process
          </p>
          {!status.memory.enabled && (
            <p className="text-xs text-yellow-400 mt-2">
              Set SUPABASE_URL and SUPABASE_ANON_KEY to persist conversations to
              the memories table.
            </p>
          )}
        </Card>

        <Card title="NanEcho training">
          {!t ? (
            <p className="text-sm opacity-70">
              No training_summary.json found under .training-progress/artifacts.
            </p>
          ) : (
            <dl className="grid grid-cols-2 gap-y-2 text-sm">
              <dt className="opacity-70">workflow run</dt>
              <dd className="font-mono">{t.workflowRun ?? "—"}</dd>
              <dt className="opacity-70">completed</dt>
              <dd className="font-mono text-xs">
                {t.completedAt ? new Date(t.completedAt).toLocaleString() : "—"}
              </dd>
              <dt className="opacity-70">best val loss</dt>
              <dd
                className={`font-mono ${t.bestValLoss !== null && t.bestValLoss > 5 ? "text-yellow-400" : "text-green-400"}`}
              >
                {t.bestValLoss?.toFixed(3) ?? "—"}
              </dd>
              <dt className="opacity-70">best quality</dt>
              <dd className="font-mono">
                {t.bestQualityScore?.toFixed(0) ?? "—"}
              </dd>
              <dt className="opacity-70">checkpoints</dt>
              <dd className="font-mono">{t.totalCheckpoints ?? "—"}</dd>
              <dt className="opacity-70">best iteration</dt>
              <dd className="font-mono">{t.bestIteration ?? "—"}</dd>
            </dl>
          )}
          {t?.bestValLoss !== null &&
            t?.bestValLoss !== undefined &&
            t.bestValLoss > 5 && (
              <p className="text-xs text-yellow-400 mt-3">
                val loss above 5 — the model has not converged. Watch this
                number fall across runs to know training is improving.
              </p>
            )}
        </Card>

        <Card title="How to read this page">
          <ul className="text-sm space-y-1 list-disc pl-5 opacity-80">
            <li>
              <b>Providers</b> turn green after the first successful chat
              response; red shows the last error so failed calls are never
              silent.
            </li>
            <li>
              <b>Memory</b> counts rows tagged{" "}
              <span className="font-mono">dte-chat</span> in Supabase; it should
              grow by two per exchange.
            </li>
            <li>
              <b>Training</b> mirrors the latest CI artifact; a falling val loss
              is the signal that NanEcho is learning.
            </li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
