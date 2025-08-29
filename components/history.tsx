"use client"

import { useEffect, useMemo, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { getConversationHistory } from "@/lib/api"
import type { ConversationHistoryResponse } from "@/types"
import { History, Search, List, Loader2 } from "lucide-react"

export default function ConversationHistory() {
  const { toast } = useToast()
  const [savedSessions, setSavedSessions] = useState<string[]>([])
  const [selectedSession, setSelectedSession] = useState<string>("")
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<ConversationHistoryResponse | null>(null)
  const [manualSession, setManualSession] = useState("")

  useEffect(() => {
    const all = JSON.parse(localStorage.getItem("qa_all_sessions") || "[]") as string[]
    setSavedSessions(all.filter(Boolean))
  }, [])

  const hasSessions = useMemo(() => savedSessions.length > 0, [savedSessions])

  async function loadSession(sessionId: string) {
    if (!sessionId) return
    setLoading(true)
    setSelectedSession(sessionId)
    setData(null)
    try {
      const resp = await getConversationHistory(sessionId)
      setData(resp)
      if (!savedSessions.includes(sessionId)) {
        const next = [...savedSessions, sessionId]
        setSavedSessions(next)
        localStorage.setItem("qa_all_sessions", JSON.stringify(next))
      }
    } catch (e: any) {
      toast({
        title: "Failed to load conversation",
        description: e?.message || "Unknown error",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  function removeSession(id: string) {
    const next = savedSessions.filter((s) => s !== id)
    setSavedSessions(next)
    localStorage.setItem("qa_all_sessions", JSON.stringify(next))
    if (selectedSession === id) {
      setSelectedSession("")
      setData(null)
    }
  }

  return (
    <div className="flex flex-col gap-8">
      <Card className="bg-white/5 text-white border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Session History
          </CardTitle>
          <CardDescription className="text-white/70">
            Explore saved sessions and view full Q&amp;A transcripts from the backend.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-3 md:grid-cols-[1fr_auto]">
            <Input
              value={manualSession}
              onChange={(e) => setManualSession(e.target.value)}
              placeholder="Paste a session ID to load..."
              className="bg-black/30 border-white/10 text-white"
            />
            <Button
              onClick={() => loadSession(manualSession.trim())}
              disabled={!manualSession.trim() || loading}
              className="justify-center"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" /> Load Session
                </>
              )}
            </Button>
          </div>

          <Separator className="bg-white/10" />

          <div>
            <div className="flex items-center gap-2 mb-2">
              <List className="h-4 w-4" />
              <div className="text-sm text-white/80">Saved Sessions</div>
              <Badge variant="secondary" className="text-black">
                {savedSessions.length}
              </Badge>
            </div>
            {!hasSessions ? (
              <div className="text-white/60">No saved sessions found. Ask a question to create one.</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {savedSessions.map((id) => (
                  <div key={id} className="flex items-center gap-2">
                    <Button
                      variant={selectedSession === id ? "default" : "outline"}
                      className={selectedSession === id ? "" : "border-white/20 text-white bg-transparent"}
                      onClick={() => loadSession(id)}
                      size="sm"
                    >
                      {id.slice(0, 8)}...
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-white/70 hover:text-white"
                      onClick={() => removeSession(id)}
                      aria-label={`Remove session ${id}`}
                    >
                      {"✕"}
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {data && (
        <Card className="bg-white/5 text-white border-white/10">
          <CardHeader>
            <CardTitle>Session {data.session_id.slice(0, 8)}...</CardTitle>
            <CardDescription className="text-white/70">{data.total_interactions} total interactions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Metric label="Interactions" value={String(data.total_interactions)} />
              <Metric
                label="Start Time"
                value={data.history[0]?.timestamp ? new Date(data.history[0].timestamp).toLocaleString() : "—"}
              />
              <Metric
                label="End Time"
                value={
                  data.history[data.history.length - 1]?.timestamp
                    ? new Date(data.history[data.history.length - 1].timestamp).toLocaleString()
                    : "—"
                }
              />
              <Metric
                label="Avg Confidence"
                value={
                  data.history.length > 0
                    ? (data.history.reduce((a, b) => a + (b.confidence ?? 0), 0) / data.history.length).toFixed(2)
                    : "0.00"
                }
              />
            </div>

            <Separator className="bg-white/10" />

            <ul className="space-y-4">
              {data.history.map((item, idx) => (
                <li key={idx} className="space-y-2">
                  <div className="text-sm text-white/60">{new Date(item.timestamp).toLocaleString()}</div>
                  <div>
                    <span className="font-semibold">Q:</span> <span className="text-white/90">{item.question}</span>
                  </div>
                  <div>
                    <span className="font-semibold">A:</span> <span className="text-white/90">{item.answer}</span>
                  </div>
                  <Badge className="bg-emerald-600">{`Confidence: ${item.confidence?.toFixed(2)}`}</Badge>
                  <Separator className="bg-white/10" />
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Card className="bg-white/5 border-white/10">
      <CardContent className="p-4">
        <div className="text-xs text-white/60">{label}</div>
        <div className="text-2xl font-semibold">{value}</div>
      </CardContent>
    </Card>
  )
}
