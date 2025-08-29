"use client"

import { useEffect, useMemo, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { getConversationHistory } from "@/lib/api"
import type { ConversationHistoryResponse } from "@/types"

export default function ConversationHistory() {
  const [allSessions, setAllSessions] = useState<string[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)

  const [historyData, setHistoryData] = useState<ConversationHistoryResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const cur = localStorage.getItem("qa_current_session_id") || ""
    const all = JSON.parse(localStorage.getItem("qa_all_sessions") || "[]") as string[]
    const combined = Array.from(new Set([...all, cur].filter(Boolean)))
    setAllSessions(combined)
    setCurrentSessionId(cur || null)
    setSelectedSession(cur || combined[0] || null)
  }, [])

  useEffect(() => {
    ;(async () => {
      if (!selectedSession) return
      try {
        setLoading(true)
        const resp = await getConversationHistory(selectedSession)
        setHistoryData(resp)
      } finally {
        setLoading(false)
      }
    })()
  }, [selectedSession])

  const options = useMemo(() => {
    return allSessions.map((s) => {
      const isCurrent = currentSessionId && s === currentSessionId
      const label = `${isCurrent ? "🟢 Current" : "📝 Session"}: ${s.slice(0, 8)}...`
      return { value: s, label }
    })
  }, [allSessions, currentSessionId])

  return (
    <div className="flex flex-col gap-6">
      <Card className="bg-white/5 text-white border-white/10">
        <CardHeader>
          <CardTitle>Conversation History</CardTitle>
          <CardDescription className="text-white/70">
            Review your past questions and answers across sessions.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {options.length === 0 ? (
            <div className="text-white/70">
              No conversation sessions found. Start a conversation in the Upload &amp; Q&amp;A tab.
            </div>
          ) : (
            <>
              <div className="grid gap-3 md:grid-cols-3">
                <div className="md:col-span-2">
                  <label className="text-sm text-white/80">Choose a session</label>
                  <Select value={selectedSession || ""} onValueChange={setSelectedSession}>
                    <SelectTrigger className="bg-black/30 border-white/10 text-white">
                      <SelectValue placeholder="Select a session" />
                    </SelectTrigger>
                    <SelectContent>
                      {options.map((o) => (
                        <SelectItem key={o.value} value={o.value}>
                          {o.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-end">
                  <Card className="bg-white/5 border-white/10 w-full">
                    <CardContent className="p-4">
                      <div className="text-xs text-white/60">Total Sessions</div>
                      <div className="text-2xl font-semibold">{options.length}</div>
                    </CardContent>
                  </Card>
                </div>
              </div>

              {loading ? (
                <div className="text-white/70">Loading session...</div>
              ) : (
                historyData && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-3 gap-3">
                      <Card className="bg-white/5 border-white/10">
                        <CardContent className="p-4">
                          <div className="text-xs text-white/60">Session</div>
                          <div className="text-xl font-semibold">{historyData.session_id.slice(0, 8)}...</div>
                        </CardContent>
                      </Card>
                      <Card className="bg-white/5 border-white/10">
                        <CardContent className="p-4">
                          <div className="text-xs text-white/60">Total Interactions</div>
                          <div className="text-2xl font-semibold">{historyData.total_interactions}</div>
                        </CardContent>
                      </Card>
                      <Card className="bg-white/5 border-white/10">
                        <CardContent className="p-4">
                          <div className="text-xs text-white/60">Status</div>
                          <div className="text-xl font-semibold">
                            {currentSessionId && selectedSession === currentSessionId ? "🟢 Active" : "📝 Archived"}
                          </div>
                        </CardContent>
                      </Card>
                    </div>

                    {historyData.history.length > 0 ? (
                      <div>
                        <h3 className="text-lg font-semibold mb-2">Conversation</h3>
                        <Accordion type="single" collapsible className="w-full">
                          {[...historyData.history].reverse().map((item, idx) => (
                            <AccordionItem key={idx} value={`h-${idx}`} className="border-white/10">
                              <AccordionTrigger className="text-left">
                                {`Q${idx + 1}: ${item.question.slice(0, 60)}...`}
                              </AccordionTrigger>
                              <AccordionContent className="text-sm text-white/90 space-y-2">
                                <div>
                                  <span className="font-semibold">Question:</span> {item.question}
                                </div>
                                <div>
                                  <span className="font-semibold">Answer:</span> {item.answer}
                                </div>
                                <div className="flex items-center gap-3">
                                  <Badge variant="secondary" className="text-black">
                                    Confidence: {item.confidence.toFixed(3)}
                                  </Badge>
                                  <Badge variant="outline" className="text-white border-white/20">
                                    {item.timestamp.slice(0, 19)}
                                  </Badge>
                                </div>
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </div>
                    ) : (
                      <div className="text-white/70">No interactions found in this session.</div>
                    )}
                  </div>
                )
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
