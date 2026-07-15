"use client";

import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
  ComposedChart, Line, Area, Cell, Scatter, ScatterChart, ZAxis,
  PieChart, Pie, Legend,
} from "recharts";
import { usd, num, pct, shortMonth } from "@/lib/format";

const AXIS = { fontSize: 12, fill: "#5b6676" };
const GRID = "#eef1f6";
const SEG_ORDER = ["VIP Champions", "Loyal Regulars", "High-Value Growers", "Occasional Buyers", "At-Risk / Dormant"];
const segColor = (name: string) => SEG_COLORS[Math.max(0, SEG_ORDER.indexOf(name)) % SEG_COLORS.length];

/* ---------- Segments: revenue by segment (interactive) ---------- */
type Seg = { name: string; count: number; revenue: number; revenue_pct: number;
  avg_recency: number; avg_frequency: number; avg_monetary: number };

const SEG_COLORS = ["#0fa3a3", "#4f9d8b", "#6b8fc9", "#c9a26b", "#d1637a"];

function SegTip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const s: Seg = payload[0].payload;
  return (
    <div className="tt">
      <div className="t">{s.name}</div>
      <div className="r">{usd(s.revenue)} · {pct(s.revenue_pct)} of revenue</div>
      <div className="r">{num(s.count)} customers · avg {usd(s.avg_monetary)}</div>
      <div className="r">recency {Math.round(s.avg_recency)}d · {s.avg_frequency.toFixed(1)} orders</div>
    </div>
  );
}

export function SegmentChart({ segments }: { segments: Seg[] }) {
  const data = [...segments].sort((a, b) => b.revenue - a.revenue);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
        <CartesianGrid horizontal={false} stroke={GRID} />
        <XAxis type="number" tick={AXIS} tickFormatter={(v) => usd(v, true)} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="name" tick={AXIS} width={128} axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: "rgba(15,163,163,0.06)" }} content={<SegTip />} />
        <Bar dataKey="revenue" radius={[0, 6, 6, 0]} maxBarSize={30} isAnimationActive={false}>
          {data.map((_, i) => <Cell key={i} fill={SEG_COLORS[i % SEG_COLORS.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ---------- Monthly revenue history (interactive) ---------- */
function MonthTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tt">
      <div className="t">{shortMonth(label)}</div>
      <div className="r">{usd(payload[0].value)}</div>
    </div>
  );
}

export function MonthlyChart({ data }: { data: { month: string; revenue: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={230}>
      <BarChart data={data} margin={{ left: 4, right: 8 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="month" tick={AXIS} tickFormatter={shortMonth} axisLine={false} tickLine={false} />
        <YAxis tick={AXIS} tickFormatter={(v) => usd(v, true)} axisLine={false} tickLine={false} width={52} />
        <Tooltip cursor={{ fill: "rgba(15,163,163,0.06)" }} content={<MonthTip />} />
        <Bar dataKey="revenue" fill="#0fa3a3" radius={[5, 5, 0, 0]} maxBarSize={38} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ---------- Forecast: history + 30-day forecast with 95% band ---------- */
function FcTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tt">
      <div className="t">{label}</div>
      {p.actual != null && <div className="r">actual: {usd(p.actual)}</div>}
      {p.forecast != null && <div className="r">forecast: {usd(p.forecast)}</div>}
      {p.band && <div className="r">95% range: {usd(p.band[0])} – {usd(p.band[1])}</div>}
    </div>
  );
}

export function ForecastChart({ series }: { series: any }) {
  const data = [
    ...series.history.map((h: any) => ({ date: h.date, actual: h.actual })),
    ...series.forecast.map((f: any) => ({ date: f.date, forecast: f.forecast, band: [f.lo, f.hi] })),
  ];
  const fmt = (d: string) => {
    const dt = new Date(d);
    return `${dt.toLocaleString("en-US", { month: "short" })} ${dt.getDate()}`;
  };
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={data} margin={{ left: 4, right: 12 }}>
        <CartesianGrid vertical={false} stroke={GRID} />
        <XAxis dataKey="date" tick={AXIS} tickFormatter={fmt} minTickGap={44} axisLine={false} tickLine={false} />
        <YAxis tick={AXIS} tickFormatter={(v) => usd(v, true)} axisLine={false} tickLine={false} width={52} />
        <Tooltip content={<FcTip />} />
        <Area dataKey="band" stroke="none" fill="#0fa3a3" fillOpacity={0.12} connectNulls isAnimationActive={false} />
        <Line dataKey="actual" stroke="#1a2233" strokeWidth={1.6} dot={false} connectNulls isAnimationActive={false} />
        <Line dataKey="forecast" stroke="#0fa3a3" strokeWidth={2} strokeDasharray="5 4" dot={false} connectNulls isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ---------- Churn drivers (SHAP) ---------- */
function ShapTip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tt">
      <div className="t">{label}</div>
      <div className="r">mean |SHAP|: {payload[0].value.toFixed(3)}</div>
    </div>
  );
}

export function ShapChart({ features }: { features: { feature: string; mean_abs_shap: number }[] }) {
  const data = features.slice(0, 6).map((f) => ({
    feature: f.feature.replace(/_/g, " "), v: f.mean_abs_shap,
  }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
        <CartesianGrid horizontal={false} stroke={GRID} />
        <XAxis type="number" tick={AXIS} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="feature" tick={AXIS} width={112} axisLine={false} tickLine={false} />
        <Tooltip cursor={{ fill: "rgba(15,163,163,0.06)" }} content={<ShapTip />} />
        <Bar dataKey="v" fill="#6b8fc9" radius={[0, 6, 6, 0]} maxBarSize={22} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ---------- Segment revenue donut ---------- */
export function SegmentDonut({ segments }: { segments: Seg[] }) {
  const data = [...segments].sort((a, b) => b.revenue - a.revenue);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <PieChart>
        <Pie data={data} dataKey="revenue" nameKey="name" cx="50%" cy="50%"
          innerRadius={58} outerRadius={92} paddingAngle={2} stroke="none" isAnimationActive={false}>
          {data.map((s) => <Cell key={s.name} fill={segColor(s.name)} />)}
        </Pie>
        <Tooltip content={<SegTip />} />
        <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
      </PieChart>
    </ResponsiveContainer>
  );
}

/* ---------- CLV vs Recency scatter (by segment) ---------- */
type Pt = { recency: number; monetary: number; frequency: number; segment: string };

function ScatterTip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const p: Pt = payload[0].payload;
  return (
    <div className="tt">
      <div className="t">{p.segment}</div>
      <div className="r">value {usd(p.monetary)} · {p.frequency} orders</div>
      <div className="r">{p.recency} days since last purchase</div>
    </div>
  );
}

export function ClvScatter({ points }: { points: Pt[] }) {
  const bySeg = SEG_ORDER.map((seg) => ({ seg, data: points.filter((p) => p.segment === seg) }))
    .filter((s) => s.data.length);
  const ys = points.map((p) => p.monetary).filter((v) => v > 0);
  const yMin = Math.max(1, Math.floor(Math.min(...ys)));
  const yMax = Math.ceil(Math.max(...ys));
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 20 }}>
        <CartesianGrid stroke={GRID} />
        <XAxis type="number" dataKey="recency" name="Recency" tick={AXIS} axisLine={false} tickLine={false}
          label={{ value: "Days since last purchase", position: "insideBottom", offset: -12, fontSize: 12, fill: "#5b6676" }} />
        <YAxis type="number" dataKey="monetary" name="Value" scale="log" domain={[yMin, yMax]}
          allowDataOverflow tick={AXIS} axisLine={false} tickLine={false}
          tickFormatter={(v) => usd(v, true)} width={54} />
        <ZAxis range={[26, 26]} />
        <Tooltip content={<ScatterTip />} cursor={{ strokeDasharray: "3 3" }} />
        {bySeg.map((s) => (
          <Scatter key={s.seg} name={s.seg} data={s.data} fill={segColor(s.seg)} fillOpacity={0.55} isAnimationActive={false} />
        ))}
        <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
