"use client"

import { Card, CardContent } from "@/components/ui/card"

type Props = {
  f1: number[]
  semantic: number[]
}

/**
 * Lightweight bucketed bar charts without external deps.
 * Buckets: [0.0-0.1), [0.1-0.2), ... [0.9-1.0]
 */
export function ScoresBarCharts({ f1, semantic }: Props) {
  const f1Buckets = bucketize(f1)
  const semBuckets = bucketize(semantic)
  const maxCount = Math.max(...f1Buckets, ...semBuckets, 1)

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Chart title="F1 Score Distribution" buckets={f1Buckets} maxCount={maxCount} color="bg-emerald-500" />
      <Chart title="Semantic Score Distribution" buckets={semBuckets} maxCount={maxCount} color="bg-purple-500" />
    </div>
  )
}

function Chart({
  title,
  buckets,
  maxCount,
  color,
}: {
  title: string
  buckets: number[]
  maxCount: number
  color: string
}) {
  return (
    <Card className="bg-white/5 text-white border-white/10">
      <CardContent className="p-4">
        <div className="text-sm mb-3">{title}</div>
        <div className="flex items-end gap-1 h-32">
          {buckets.map((count, idx) => {
            const height = Math.round((count / Math.max(maxCount, 1)) * 100)
            return (
              <div key={idx} className="flex-1 flex flex-col items-center">
                <div className={`w-full ${color}`} style={{ height: `${height}%` }} aria-label={`${count} items`} />
              </div>
            )
          })}
        </div>
        <div className="flex justify-between mt-2 text-xs text-white/60">
          <span>{"0.0"}</span>
          <span>{"0.5"}</span>
          <span>{"1.0"}</span>
        </div>
      </CardContent>
    </Card>
  )
}

function bucketize(values: number[]) {
  const buckets = new Array(10).fill(0)
  for (const v of values) {
    const clamped = Math.max(0, Math.min(0.999999, v || 0))
    const idx = Math.floor(clamped * 10)
    buckets[idx]++
  }
  return buckets
}
