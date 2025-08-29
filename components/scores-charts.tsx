"use client"

import { BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Bar, ResponsiveContainer } from "recharts"

export function ScoresBarCharts({ f1, semantic }: { f1: number[]; semantic: number[] }) {
  const f1Data = f1.map((v, i) => ({ idx: i + 1, score: Number(v.toFixed(3)) }))
  const semData = semantic.map((v, i) => ({ idx: i + 1, score: Number(v.toFixed(3)) }))
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="h-64 w-full bg-white/5 rounded-lg p-3">
        <div className="text-sm mb-2">F1 Scores Distribution</div>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={f1Data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="idx" stroke="#fff" />
            <YAxis domain={[0, 1]} stroke="#fff" />
            <Tooltip
              contentStyle={{ background: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)", color: "#fff" }}
            />
            <Bar dataKey="score" fill="#22c55e" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="h-64 w-full bg-white/5 rounded-lg p-3">
        <div className="text-sm mb-2">Semantic Scores Distribution</div>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={semData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="idx" stroke="#fff" />
            <YAxis domain={[0, 1]} stroke="#fff" />
            <Tooltip
              contentStyle={{ background: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)", color: "#fff" }}
            />
            <Bar dataKey="score" fill="#06b6d4" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
