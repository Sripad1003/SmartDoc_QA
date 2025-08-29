"use client"

import type {
  AskResponse,
  DocumentListResponse,
  EvaluateResponse,
  GenerateQuestionsResponse,
  HealthResponse,
  UploadResponse,
} from "@/types"

// Frontend talks to Next.js proxy routes to avoid CORS.
const NEXT_API = "/api"

export async function getHealth(): Promise<HealthResponse> {
  const r = await fetch(`${NEXT_API}/health`, { method: "GET" })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getDocuments(): Promise<DocumentListResponse> {
  const r = await fetch(`${NEXT_API}/list-documents`, { method: "GET" })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function uploadDocuments(files: FileList): Promise<UploadResponse> {
  const formData = new FormData()
  Array.from(files).forEach((f) => formData.append("files", f, f.name))
  const r = await fetch(`${NEXT_API}/upload-documents`, {
    method: "POST",
    body: formData,
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function askQuestion(question: string, sessionId?: string): Promise<AskResponse> {
  const r = await fetch(`${NEXT_API}/ask-question`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId || null }),
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function generateQuestions(
  docId: string,
  maxQuestions = 8,
  seed?: number,
): Promise<GenerateQuestionsResponse> {
  const url = new URL(`${NEXT_API}/generate-questions/${docId}`, globalThis.location?.origin || "http://localhost")
  url.searchParams.set("max_questions", String(maxQuestions))
  if (seed) url.searchParams.set("seed", String(seed))
  const r = await fetch(url.toString().replace(url.origin, ""), { method: "GET" })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function evaluateDocument(docId: string): Promise<EvaluateResponse> {
  const r = await fetch(`${NEXT_API}/evaluate/${docId}`, { method: "GET" })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getConversationHistory(sessionId: string): Promise<ConversationHistoryResponse> {
  const r = await fetch(`${NEXT_API}/conversation-history/${sessionId}`, { method: "GET" })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export type ConversationHistoryResponse = {
  session_id: string
  history: {
    question: string
    answer: string
    confidence: number
    timestamp: string
  }[]
  total_interactions: number
}
