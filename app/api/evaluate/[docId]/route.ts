import { NextResponse } from "next/server"
import { getApiBase } from "@/lib/server-config"

export async function GET(_req: Request, { params }: { params: { docId: string } }) {
  const API = getApiBase()
  const r = await fetch(`${API}/evaluate/${params.docId}`, { method: "GET" })
  const text = await r.text()
  return new NextResponse(text, {
    status: r.status,
    headers: { "content-type": r.headers.get("content-type") ?? "application/json" },
  })
}
