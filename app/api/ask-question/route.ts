import { NextResponse } from "next/server"
import { getApiBase } from "@/lib/server-config"

export async function POST(req: Request) {
  const API = getApiBase()
  const payload = await req.json()
  const r = await fetch(`${API}/ask-question`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  })
  const text = await r.text()
  return new NextResponse(text, {
    status: r.status,
    headers: { "content-type": r.headers.get("content-type") ?? "application/json" },
  })
}
