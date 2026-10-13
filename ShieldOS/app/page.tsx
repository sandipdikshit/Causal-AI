"use client";

import {
  Activity,
  ArrowRight,
  BadgeCheck,
  Ban,
  Check,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  Cpu,
  Fingerprint,
  Gauge,
  GitBranch,
  LockKeyhole,
  Play,
  Radio,
  RefreshCcw,
  ScanLine,
  ShieldCheck,
  ShieldEllipsis,
  Smartphone,
  Sparkles,
  TimerReset,
  TriangleAlert,
  Zap,
} from "lucide-react";
import { useMemo, useState } from "react";

type TransactionInput = {
  amount: number;
  velocity: number;
  deviceTrust: number;
  simSwap: boolean;
};

type Decision = "APPROVE" | "STEP-UP CHALLENGE" | "HARD BLOCK";

type RiskEvaluation = {
  score: number;
  action: Decision;
  pattern: string;
  amountRisk: number;
  velocityRisk: number;
  deviceRisk: number;
  simRisk: number;
};

type AuditRow = {
  id: string;
  timestamp: string;
  pattern: string;
  score: number;
  action: Decision;
};

const clamp = (value: number, min = 0, max = 100) =>
  Math.min(max, Math.max(min, value));

function evaluateRisk(input: TransactionInput): RiskEvaluation {
  const amountSignal = clamp((Math.log10(Math.max(input.amount, 1)) - 1.8) * 30);
  const velocitySignal = clamp((input.velocity / 24) * 100);
  const deviceSignal = 100 - input.deviceTrust;
  const simSignal = input.simSwap ? 100 : 0;

  const amountRisk = amountSignal * 0.3;
  const velocityRisk = velocitySignal * 0.3;
  const deviceRisk = deviceSignal * 0.25;
  const simRisk = simSignal * 0.15;
  const score = Math.round(clamp(amountRisk + velocityRisk + deviceRisk + simRisk));

  let action: Decision = "APPROVE";
  if (score > 75 || input.simSwap) action = "HARD BLOCK";
  else if (score >= 40) action = "STEP-UP CHALLENGE";

  let pattern = "Clean";
  if (input.simSwap) pattern = "SIM Swap Anomaly";
  else if (input.deviceTrust < 35) pattern = "ATO Attempt";
  else if (input.velocity >= 15) pattern = "Velocity Burst";
  else if (input.amount >= 15000) pattern = "High-Value Irregularity";

  return { score, action, pattern, amountRisk, velocityRisk, deviceRisk, simRisk };
}

const initialLogs: AuditRow[] = [
  { id: "TXN-93A7C2", timestamp: "09:42:18.024", pattern: "SIM Swap Anomaly", score: 89, action: "HARD BLOCK" },
  { id: "TXN-5E18B4", timestamp: "09:41:52.913", pattern: "Clean", score: 12, action: "APPROVE" },
  { id: "TXN-42FD91", timestamp: "09:41:08.441", pattern: "Velocity Burst", score: 63, action: "STEP-UP CHALLENGE" },
  { id: "TXN-A1047D", timestamp: "09:40:37.118", pattern: "ATO Attempt", score: 81, action: "HARD BLOCK" },
  { id: "TXN-8CC32E", timestamp: "09:39:54.602", pattern: "Clean", score: 18, action: "APPROVE" },
];

const metrics = [
  { label: "Total volume monitored", value: "$12,450,200", meta: "+8.2% vs. prior window", icon: CircleDollarSign, trend: [7, 10, 9, 15, 13, 18, 19, 24, 22, 29] },
  { label: "Detection accuracy rate", value: "98.4%", meta: "Model ensemble · v4.8", icon: BadgeCheck, trend: [14, 15, 15, 17, 17, 18, 20, 20, 22, 23] },
  { label: "False positive ratio", value: "3.1:1", meta: "↓ 0.4 from last 24h", icon: GitBranch, trend: [25, 23, 24, 19, 20, 17, 15, 16, 12, 10] },
  { label: "Median processing latency", value: "24ms", meta: "p99 · 61ms", icon: Clock3, trend: [18, 15, 17, 13, 12, 14, 10, 11, 8, 9] },
];

const actionConfig = {
  APPROVE: { className: "decision-approve", compactClass: "compact-approve", Icon: Check, description: "Zero-friction checkout path", route: "Settlement rail · instant" },
  "STEP-UP CHALLENGE": { className: "decision-stepup", compactClass: "compact-stepup", Icon: ShieldEllipsis, description: "Trigger asynchronous biometric / MFA validation", route: "Identity challenge · async" },
  "HARD BLOCK": { className: "decision-block", compactClass: "compact-block", Icon: Ban, description: "Route to Fraud Operations manual review queue", route: "Fraud Ops · priority P1" },
};

function Sparkline({ values }: { values: number[] }) {
  const points = values.map((value, index) => `${index * 10},${32 - value}`).join(" ");
  const gradientId = `spark-${values[0]}`;
  return (
    <svg viewBox="0 0 90 34" className="h-9 w-[90px]" aria-hidden="true">
      <defs><linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#53e6b1" stopOpacity=".28" /><stop offset="100%" stopColor="#53e6b1" stopOpacity="0" /></linearGradient></defs>
      <polygon points={`0,34 ${points} 90,34`} fill={`url(#${gradientId})`} />
      <polyline points={points} fill="none" stroke="#53e6b1" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function RangeControl({ label, eyebrow, value, min, max, step = 1, unit, onChange }: { label: string; eyebrow: string; value: number; min: number; max: number; step?: number; unit: string; onChange: (value: number) => void; }) {
  const progress = ((value - min) / (max - min)) * 100;
  return (
    <div className="control-block">
      <div className="flex items-end justify-between gap-4">
        <div><p className="control-eyebrow">{eyebrow}</p><label className="control-label">{label}</label></div>
        <div className="value-field">
          <input aria-label={label} type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(clamp(Number(event.target.value), min, max))} />
          <span>{unit}</span>
        </div>
      </div>
      <input className="risk-slider mt-5" aria-label={`${label} slider`} type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ "--progress": `${progress}%` } as React.CSSProperties} />
      <div className="mt-2 flex justify-between text-[10px] font-medium text-slate-600"><span>{min.toLocaleString()}</span><span>{max.toLocaleString()}</span></div>
    </div>
  );
}

function ContributionBar({ label, value, weight }: { label: string; value: number; weight: string; }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-[11px]"><span className="font-medium text-slate-400">{label}</span><span className="font-mono text-slate-300">+{value.toFixed(1)} <span className="text-slate-600">· {weight}</span></span></div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.05]"><div className="h-full rounded-full bg-gradient-to-r from-emerald-500/70 to-cyan-400/80 transition-all duration-500" style={{ width: `${Math.min(100, value * 3.3)}%` }} /></div>
    </div>
  );
}

export default function Home() {
  const [inputs, setInputs] = useState<TransactionInput>({ amount: 8750, velocity: 9, deviceTrust: 58, simSwap: false });
  const [logs, setLogs] = useState<AuditRow[]>(initialLogs);
  const [processing, setProcessing] = useState(false);
  const [runCount, setRunCount] = useState(0);
  const [selectedTrace, setSelectedTrace] = useState<string | null>(null);
  const risk = useMemo(() => evaluateRisk(inputs), [inputs]);
  const decision = actionConfig[risk.action];

  const updateInput = <K extends keyof TransactionInput>(key: K, value: TransactionInput[K]) => setInputs((current) => ({ ...current, [key]: value }));

  const executePipeline = () => {
    if (processing) return;
    setProcessing(true);
    window.setTimeout(() => {
      const now = new Date();
      const serial = (924801 + runCount * 731).toString(16).toUpperCase().slice(-6);
      setLogs((current) => [{ id: `TXN-${serial.padStart(6, "0")}`, timestamp: now.toLocaleTimeString("en-GB", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }), pattern: risk.pattern, score: risk.score, action: risk.action }, ...current.slice(0, 6)]);
      setRunCount((count) => count + 1);
      setProcessing(false);
    }, 720);
  };

  const loadScenario = (scenario: "clean" | "velocity" | "takeover") => {
    const presets = {
      clean: { amount: 129, velocity: 2, deviceTrust: 94, simSwap: false },
      velocity: { amount: 4200, velocity: 19, deviceTrust: 61, simSwap: false },
      takeover: { amount: 18500, velocity: 21, deviceTrust: 18, simSwap: true },
    };
    setInputs(presets[scenario]);
  };

  const navigateTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const exportAudit = () => {
    const header = ["Transaction ID", "Timestamp", "Attack Pattern", "Risk Score", "Settlement Action"];
    const rows = logs.map((log) => [log.id, `${log.timestamp} UTC`, log.pattern, String(log.score), log.action]);
    const csv = [header, ...rows]
      .map((row) => row.map((value) => `"${value.replaceAll('"', '""')}"`).join(","))
      .join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "shieldos-traceability-audit.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="min-h-screen overflow-hidden bg-[#06090e] text-slate-100">
      <div className="ambient-orb ambient-orb-one" /><div className="ambient-orb ambient-orb-two" /><div className="dashboard-grid" />
      <div className="relative z-10 mx-auto max-w-[1540px] px-4 pb-10 pt-4 sm:px-6 lg:px-8">
        <header className="glass-shell flex min-h-[68px] items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="brand-mark"><ShieldCheck size={21} strokeWidth={2.1} /><span className="brand-pulse" /></div>
            <div><div className="flex items-center gap-2"><h1 className="text-[17px] font-semibold tracking-[-0.025em] text-white">ShieldOS</h1><span className="rounded-full border border-emerald-400/20 bg-emerald-400/[0.08] px-2 py-0.5 text-[9px] font-bold tracking-[0.16em] text-emerald-300">LIVE</span></div><p className="mt-0.5 hidden text-[11px] text-slate-500 sm:block">Real-Time Fraud &amp; Transaction Decisioning Engine</p></div>
          </div>
          <nav className="hidden items-center gap-7 text-[11px] font-medium text-slate-500 lg:flex" aria-label="Dashboard sections">
            <button onClick={() => navigateTo("command-center")} className="flex items-center gap-2 text-slate-300 transition hover:text-white"><Activity size={14} className="text-emerald-400" /> Decisioning</button>
            <button onClick={() => navigateTo("risk-engine")} className="transition hover:text-slate-200">Model registry</button>
            <button onClick={() => navigateTo("simulator")} className="transition hover:text-slate-200">Policy studio</button>
            <button onClick={() => navigateTo("audit-log")} className="transition hover:text-slate-200">Investigations</button>
          </nav>
          <div className="flex items-center gap-3"><div className="hidden items-center gap-2 rounded-full border border-white/[0.06] bg-white/[0.025] px-3 py-2 text-[10px] text-slate-400 sm:flex"><Radio size={12} className="text-emerald-400" /><span>eu-west-1</span><span className="h-1 w-1 rounded-full bg-slate-700" /><span>Healthy</span></div><button onClick={() => navigateTo("audit-log")} aria-label="Open operator investigations" className="flex h-9 w-9 items-center justify-center rounded-full border border-white/[0.08] bg-white/[0.04] text-slate-300 transition hover:border-white/15 hover:bg-white/[0.08]"><Fingerprint size={16} /></button></div>
        </header>

        <section id="command-center" className="mt-6 scroll-mt-4">
          <div className="mb-4 flex items-end justify-between gap-5">
            <div><div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-emerald-400/80"><Zap size={12} /> Decision command center</div><h2 className="text-2xl font-semibold tracking-[-0.04em] text-white sm:text-[30px]">Live transaction defense</h2><p className="mt-1 max-w-2xl text-[12px] leading-5 text-slate-500">Evaluate the event, expose every risk contribution, and route the settlement action in milliseconds.</p></div>
            <div className="hidden items-center gap-2 text-[10px] text-slate-500 md:flex"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />1,248 events / sec</div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {metrics.map((metric) => { const Icon = metric.icon; return <article key={metric.label} className="metric-card group"><div className="flex items-start justify-between gap-4"><div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500"><Icon size={14} className="text-slate-400" /> {metric.label}</div><Sparkline values={metric.trend} /></div><div className="-mt-1 text-[25px] font-semibold tracking-[-0.045em] text-white">{metric.value}</div><div className="mt-1.5 text-[10px] text-slate-600">{metric.meta}</div></article>; })}
          </div>
        </section>

        <section className="mt-4 grid gap-4 xl:grid-cols-[0.92fr_1.38fr]">
          <article id="simulator" className="panel-card scroll-mt-4 overflow-hidden">
            <div className="panel-heading"><div className="panel-icon"><ScanLine size={17} /></div><div><p className="panel-kicker">01 · Event simulation</p><h3 className="panel-title">Transaction ingestion &amp; rule simulator</h3></div><span className="ml-auto hidden rounded-md border border-white/[0.06] bg-black/20 px-2 py-1 font-mono text-[9px] text-slate-600 sm:inline">POST /v1/decision</span></div>
            <div className="border-b border-white/[0.055] px-5 py-3.5 sm:px-6"><div className="flex flex-wrap items-center gap-2"><span className="mr-1 text-[10px] font-medium text-slate-600">Load scenario</span>{[["clean", "Trusted checkout"], ["velocity", "Velocity burst"], ["takeover", "Account takeover"]].map(([key, label]) => <button key={key} onClick={() => loadScenario(key as "clean" | "velocity" | "takeover")} className="scenario-chip">{label}</button>)}</div></div>
            <div className="space-y-3 p-4 sm:p-5">
              <RangeControl eyebrow="Monetary exposure" label="Transaction amount" value={inputs.amount} min={10} max={50000} step={10} unit="USD" onChange={(value) => updateInput("amount", value)} />
              <RangeControl eyebrow="Behavioral signal" label="Historical account velocity" value={inputs.velocity} min={0} max={30} unit="TX / HR" onChange={(value) => updateInput("velocity", value)} />
              <RangeControl eyebrow="Device intelligence" label="Fingerprint trust score" value={inputs.deviceTrust} min={0} max={100} unit="/ 100" onChange={(value) => updateInput("deviceTrust", value)} />
              <div className={`control-block flex items-center justify-between gap-4 ${inputs.simSwap ? "danger-control" : ""}`}><div className="flex items-center gap-3"><div className={`signal-icon ${inputs.simSwap ? "signal-icon-danger" : ""}`}><Smartphone size={16} /></div><div><p className="control-eyebrow">Telco intelligence</p><label htmlFor="sim-swap" className="control-label">SIM swap detected</label><p className="mt-1 text-[10px] text-slate-600">Confirmed within the last 72 hours</p></div></div><button id="sim-swap" role="switch" aria-checked={inputs.simSwap} onClick={() => updateInput("simSwap", !inputs.simSwap)} className={`toggle-track ${inputs.simSwap ? "toggle-active" : ""}`}><span /></button></div>
              <button onClick={executePipeline} disabled={processing} className="execute-button group"><span className="execute-glow" />{processing ? <RefreshCcw size={16} className="animate-spin" /> : <Play size={16} fill="currentColor" />}<span>{processing ? "Executing decision graph…" : "Execute Real-Time Counter-Fraud Pipeline"}</span><ArrowRight size={15} className="ml-auto transition-transform group-hover:translate-x-0.5" /></button>
            </div>
          </article>

          <article id="risk-engine" className={`panel-card scroll-mt-4 overflow-hidden ${processing ? "panel-processing" : ""}`}>
            <div className="panel-heading"><div className="panel-icon"><Cpu size={17} /></div><div><p className="panel-kicker">02 · Decision intelligence</p><h3 className="panel-title">Causal risk engine &amp; orchestration graph</h3></div><div className="ml-auto flex items-center gap-2 text-[9px] font-medium text-slate-600"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> model v4.8.2</div></div>
            <div className="grid min-h-[506px] lg:grid-cols-[1.08fr_0.92fr]">
              <div className="border-b border-white/[0.055] p-5 lg:border-b-0 lg:border-r sm:p-6">
                <div className="mb-4 flex items-center justify-between"><div><p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">Live execution graph</p><p className="mt-1 text-[11px] text-slate-500">Signals flow top-to-bottom</p></div><span className="rounded-full border border-white/[0.06] px-2.5 py-1 font-mono text-[9px] text-slate-500">24.1 ms</span></div>
                <div className="pipeline-graph">
                  {[
                    { icon: ScanLine, step: "01", title: "Event normalization", detail: "Schema verified · 42 features", time: "2.3 ms" },
                    { icon: Fingerprint, step: "02", title: "Identity signal fusion", detail: `Device trust ${inputs.deviceTrust}/100 · telco checked`, time: "5.8 ms" },
                    { icon: GitBranch, step: "03", title: "Contextual risk scoring", detail: "Weighted evidence · calibrated", time: "11.4 ms" },
                    { icon: LockKeyhole, step: "04", title: "Policy orchestration", detail: `Route: ${decision.route}`, time: "4.6 ms" },
                  ].map((node, index) => { const Icon = node.icon; return <div key={node.step} className="relative"><div className={`pipeline-node ${processing ? "pipeline-node-active" : ""}`} style={{ animationDelay: `${index * 120}ms` }}><div className="node-icon"><Icon size={16} /></div><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><span className="font-mono text-[9px] text-emerald-400/70">{node.step}</span><h4 className="truncate text-[12px] font-medium text-slate-200">{node.title}</h4></div><p className="mt-1 truncate text-[10px] text-slate-600">{node.detail}</p></div><span className="font-mono text-[9px] text-slate-600">{node.time}</span><ChevronRight size={13} className="text-slate-700" /></div>{index < 3 && <div className="pipeline-connector"><span /></div>}</div>; })}
                </div>
                <div className="mt-5 rounded-xl border border-white/[0.055] bg-black/20 p-4"><div className="mb-3 flex items-center justify-between"><span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">Weighted evidence</span><span className="font-mono text-[9px] text-slate-700">Σ contribution</span></div><div className="space-y-3"><ContributionBar label="Amount exposure" value={risk.amountRisk} weight="30%" /><ContributionBar label="Velocity pressure" value={risk.velocityRisk} weight="30%" /><ContributionBar label="Device distrust" value={risk.deviceRisk} weight="25%" /><ContributionBar label="SIM swap signal" value={risk.simRisk} weight="15%" /></div></div>
              </div>
              <div className="flex flex-col p-5 sm:p-6">
                <div className="flex items-start justify-between gap-3"><div><p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">Aggregate risk score</p><p className="mt-1 text-[10px] text-slate-600">Continuously recalculated</p></div><div className="flex items-center gap-1.5 text-[9px] text-emerald-400/80"><Sparkles size={12} /> explainable</div></div>
                <div className="mx-auto my-5"><div className="risk-ring" style={{ "--risk": `${risk.score * 3.6}deg`, "--risk-color": risk.action === "APPROVE" ? "#3edda5" : risk.action === "STEP-UP CHALLENGE" ? "#f2bc57" : "#f05675" } as React.CSSProperties}><div className="risk-ring-inner"><span className="text-[48px] font-semibold leading-none tracking-[-0.07em] text-white">{risk.score}</span><span className="mt-1 font-mono text-[9px] uppercase tracking-[0.18em] text-slate-600">of 100</span></div></div></div>
                <div className={`decision-card ${decision.className}`}><div className="flex items-center gap-2"><decision.Icon size={15} /><span className="text-[11px] font-bold tracking-[0.08em]">{risk.action}</span></div><p className="mt-2.5 text-[11px] leading-5">{decision.description}</p><div className="mt-3 flex items-center justify-between border-t border-current/10 pt-3 font-mono text-[9px] opacity-70"><span>{decision.route}</span><span>policy-eu-07</span></div></div>
                <div className="mt-auto pt-5"><div className="mb-2 flex justify-between font-mono text-[9px] text-slate-700"><span>0 · APPROVE</span><span>40 · STEP-UP</span><span>76 · BLOCK</span></div><div className="threshold-track"><div className="bg-emerald-400/70" style={{ width: "40%" }} /><div className="bg-amber-400/70" style={{ width: "36%" }} /><div className="bg-rose-500/70" style={{ width: "24%" }} /><span className="threshold-marker" style={{ left: `${risk.score}%` }} /></div><div className="mt-4 rounded-lg border border-white/[0.05] bg-white/[0.02] px-3 py-2.5 font-mono text-[9px] leading-4 text-slate-600">score = .30·amount + .30·velocity + .25·device_risk + .15·sim_swap</div></div>
              </div>
            </div>
          </article>
        </section>

        <section id="audit-log" className="panel-card mt-4 scroll-mt-4 overflow-hidden">
          <div className="panel-heading"><div className="panel-icon"><TimerReset size={17} /></div><div><p className="panel-kicker">03 · Evidence &amp; settlement</p><h3 className="panel-title">Real-time traceability audit</h3></div><div className="ml-auto hidden items-center gap-3 sm:flex"><div className="flex items-center gap-1.5 text-[9px] text-slate-600"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> append-only stream</div><button onClick={exportAudit} className="rounded-md border border-white/[0.06] bg-white/[0.025] px-2.5 py-1.5 text-[9px] font-medium text-slate-500 transition hover:text-slate-300">Export trace</button></div></div>
          <div className="overflow-x-auto"><table className="w-full min-w-[860px] border-collapse text-left"><thead><tr><th>Transaction ID</th><th>Timestamp</th><th>Vector attack pattern detected</th><th>Risk score</th><th>Final settlement action</th><th className="w-10" /></tr></thead><tbody>{logs.map((log, index) => { const config = actionConfig[log.action]; return <tr key={`${log.id}-${index}`} className={`${index === 0 && runCount > 0 ? "new-row" : ""} ${selectedTrace === log.id ? "selected-row" : ""}`}><td><span className="font-mono text-[11px] text-slate-300">{log.id}</span></td><td><span className="font-mono text-[10px] text-slate-600">{log.timestamp} UTC</span></td><td><span className="flex items-center gap-2 text-[11px] text-slate-400">{log.pattern === "Clean" ? <ShieldCheck size={13} className="text-emerald-400/70" /> : <TriangleAlert size={13} className="text-amber-400/70" />}{log.pattern}</span></td><td><div className="flex items-center gap-3"><span className="w-6 font-mono text-[11px] font-medium text-slate-300">{log.score}</span><span className="h-1 w-20 overflow-hidden rounded-full bg-white/[0.05]"><span className="block h-full rounded-full bg-slate-500" style={{ width: `${log.score}%` }} /></span></div></td><td><span className={`compact-decision ${config.compactClass}`}><config.Icon size={11} /> {log.action}</span></td><td><button onClick={() => setSelectedTrace(log.id)} aria-label={`Inspect ${log.id}`} className="text-slate-700 transition hover:text-slate-300"><ChevronRight size={15} /></button></td></tr>; })}</tbody></table></div>
          <div className="flex items-center justify-between border-t border-white/[0.05] px-5 py-3 text-[9px] text-slate-700 sm:px-6"><span>{selectedTrace ? `Trace ${selectedTrace} selected for investigation` : `Showing latest ${logs.length} decisions · retention 365 days`}</span><span className="flex items-center gap-1.5"><Gauge size={11} /> Audit stream latency 8ms</span></div>
        </section>
      </div>
    </main>
  );
}
