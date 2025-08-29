import { NextResponse } from "next/server"
import { getApiBase } from "@/lib/server-config"

export async function POST(req: Request) {
  const API = getApiBase()
  const formData = await req.formData()
  const r = await fetch(`${API}/upload-documents`, {
    method: "POST",
    body: formData,
  })
  const text = await r.text()
  return new NextResponse(text, {
    status: r.status,
    headers: { "content-type": r.headers.get("content-type") ?? "application/json" },
  })
}
