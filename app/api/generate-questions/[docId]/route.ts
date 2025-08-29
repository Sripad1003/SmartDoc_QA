import { NextResponse } from "next/server"
import { getApiBase } from "@/lib/server-config"

export async function GET(req: Request, { params }: { params: { docId: string } }) {
  const API = getApiBase()
  const { searchParams } = new URL(req.url)
  const url = new URL(`${API}/generate-questions/${params.docId}`)
  for (const [k, v] of searchParams.entries()) url.searchParams.set(k, v)
  const r = await fetch(url.toString(), { method: "GET" })
  const text = await r.text()
  return new NextResponse(text, {
    status: r.status,
    headers: { "content-type": r.headers.get("content-type") ?? "application/json" },
  })
}
