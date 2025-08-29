"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Separator } from "@/components/ui/separator"
import { useToast } from "@/hooks/use-toast"
import { Upload, Send, Sparkles, PlusCircle, FileText } from "lucide-react"
import { askQuestion, getDocuments, uploadDocuments } from "@/lib/api"
import type { AskResponse, DocumentListResponse, SourceItem } from "@/types"

export default function UploadAndQA() {
  const { toast } = useToast()

  // Documents
  const [documents, setDocuments] = useState<DocumentListResponse["documents"]>([])
  const [loadingDocs, setLoadingDocs] = useState(false)

  // Upload
  const [files, setFiles] = useState<FileList | null>(null)
  const [uploading, setUploading] = useState(false)

  // Q&A
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState<AskResponse | null>(null)
  const [asking, setAsking] = useState(false)

  // Sessions
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [allSessions, setAllSessions] = useState<string[]>([])
  const sessionsRef = useRef<{ current: string | null; all: string[] }>({ current: null, all: [] })

  const loadDocuments = useCallback(async () => {
    try {
      setLoadingDocs(true)
      const resp = await getDocuments()
      setDocuments(resp.documents)
    } catch (e) {
      toast({ title: "Failed to load documents", variant: "destructive" })
    } finally {
      setLoadingDocs(false)
    }
  }, [toast])

  // Initialize sessions from localStorage
  useEffect(() => {
    const cur = localStorage.getItem("qa_current_session_id")
    const all = JSON.parse(localStorage.getItem("qa_all_sessions") || "[]") as string[]
    setCurrentSessionId(cur)
    setAllSessions(all)
    sessionsRef.current = { current: cur, all }
    loadDocuments()
  }, [loadDocuments])

  useEffect(() => {
    localStorage.setItem("qa_current_session_id", currentSessionId || "")
  }, [currentSessionId])

  useEffect(() => {
    localStorage.setItem("qa_all_sessions", JSON.stringify(allSessions))
  }, [allSessions])

  async function handleUpload() {
    if (!files || files.length === 0) {
      toast({ title: "Select files to upload", variant: "destructive" })
      return
    }
    setUploading(true)
    try {
      const result = await uploadDocuments(files)
      const success = result.summary?.successful ?? 0
      toast({
        title: "Upload completed",
        description: `${success} of ${result.summary?.total_files ?? files.length} documents processed`,
      })
      await loadDocuments()
      setFiles(null)
    } catch (e: any) {
      toast({ title: "Upload failed", description: e?.message || "Unknown error", variant: "destructive" })
    } finally {
      setUploading(false)
    }
  }

  async function handleAsk() {
    if (!question.trim()) return
    setAsking(true)
    try {
      const resp = await askQuestion(question, currentSessionId || undefined)
      setAnswer(resp)
      // Update session state
      setCurrentSessionId(resp.session_id)
      if (!allSessions.includes(resp.session_id)) {
        const next = [...allSessions, resp.session_id]
        setAllSessions(next)
      }
      toast({ title: "Answer ready", description: "See below for details." })
    } catch (e: any) {
      toast({ title: "Request failed", description: e?.message || "Unknown error", variant: "destructive" })
    } finally {
      setAsking(false)
    }
  }

  function createNewSession() {
    if (currentSessionId && !allSessions.includes(currentSessionId)) {
      setAllSessions((prev) => [...prev, currentSessionId])
    }
    setCurrentSessionId(null)
    setAnswer(null)
    toast({ title: "New session", description: "A new session will be created with your next question." })
  }

  const docList = useMemo(() => documents, [documents])

  return (
    <div className="flex flex-col gap-8">
      {/* Documents */}
      <Card className="bg-white/5 text-white border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Your Documents
          </CardTitle>
          <CardDescription className="text-white/70">
            View all processed documents available for Q&amp;A and evaluation.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {loadingDocs ? (
            <div className="text-white/70">Loading documents...</div>
          ) : docList.length === 0 ? (
            <div className="text-white/70">No documents uploaded yet.</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {docList.map((doc) => (
                <Card key={doc.doc_id} className="bg-white/5 border-white/10">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">{doc.filename}</CardTitle>
                    <CardDescription className="text-white/60">
                      {"Processed: "}
                      {doc.processed_at}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="flex items-center justify-between">
                    <div className="flex items-center gap-3 text-sm text-white/80">
                      <Badge className="bg-emerald-600 hover:bg-emerald-700">Chunks: {doc.chunk_count}</Badge>
                      <Badge variant="secondary" className="text-black">
                        Length: {doc.text_length}
                      </Badge>
                    </div>
                    <Badge variant="outline" className="text-white border-white/20">
                      {doc.doc_id.slice(0, 8)}...
                    </Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Upload */}
      <Card className="bg-white/5 text-white border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload New Documents
          </CardTitle>
          <CardDescription className="text-white/70">Supported: PDF, DOCX, TXT, HTML, Markdown</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-[1fr_auto]">
            <div>
              <Label htmlFor="files" className="text-white">
                Choose files
              </Label>
              <Input
                id="files"
                type="file"
                multiple
                onChange={(e) => setFiles(e.target.files)}
                className="bg-black/30 border-white/10 text-white file:text-white"
                accept=".pdf,.docx,.txt,.html,.md"
              />
            </div>
            <div className="flex items-end">
              <Button
                onClick={handleUpload}
                disabled={uploading || !files || files.length === 0}
                className="w-full md:w-auto"
              >
                {uploading ? (
                  <>
                    <Sparkles className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    Upload &amp; Process
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Q&A */}
      <Card className="bg-white/5 text-white border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Send className="h-5 w-5" />
            Ask Questions
          </CardTitle>
          <CardDescription className="text-white/70">
            Session-based Q&amp;A. Maintains context across your last messages.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Session */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="text-sm">
              {currentSessionId ? (
                <span className="text-white/80">
                  {"Current Session: "}
                  <Badge variant="outline" className="text-white border-white/20">
                    {currentSessionId.slice(0, 8)}...
                  </Badge>
                </span>
              ) : (
                <span className="text-white/60">No active session</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-black">
                Sessions: {allSessions.length}
              </Badge>
              <Button
                variant="outline"
                onClick={createNewSession}
                className="border-white/20 text-white bg-transparent"
              >
                <PlusCircle className="mr-2 h-4 w-4" />
                New Session
              </Button>
            </div>
          </div>

          <Separator className="bg-white/10" />

          {/* Question input */}
          <div className="grid gap-3 md:grid-cols-[1fr_auto]">
            <Input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your question..."
              className="bg-black/30 border-white/10 text-white"
            />
            <Button onClick={handleAsk} disabled={!question || asking}>
              {asking ? (
                <>
                  <Send className="mr-2 h-4 w-4 animate-pulse" />
                  Asking...
                </>
              ) : (
                <>
                  <Send className="mr-2 h-4 w-4" />
                  Ask Question
                </>
              )}
            </Button>
          </div>

          {/* Answer */}
          {answer && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold mb-2">Answer</h3>
                <p className="text-white/90">{answer.answer}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <Card className="bg-white/5 border-white/10">
                  <CardContent className="p-4">
                    <div className="text-xs text-white/60">Confidence</div>
                    <div className="text-2xl font-semibold">{answer.confidence.toFixed(2)}</div>
                  </CardContent>
                </Card>
                <Card className="bg-white/5 border-white/10">
                  <CardContent className="p-4">
                    <div className="text-xs text-white/60">Response Time</div>
                    <div className="text-2xl font-semibold">{answer.response_time.toFixed(2)}s</div>
                  </CardContent>
                </Card>
                <Card className="bg-white/5 border-white/10">
                  <CardContent className="p-4">
                    <div className="text-xs text-white/60">Sources</div>
                    <div className="text-2xl font-semibold">{answer.sources?.length ?? 0}</div>
                  </CardContent>
                </Card>
              </div>

              {answer.sources && answer.sources.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-2">Sources</h3>
                  <Accordion type="single" collapsible className="w-full">
                    {answer.sources.map((s: SourceItem, idx: number) => (
                      <AccordionItem key={idx} value={`src-${idx}`} className="border-white/10">
                        <AccordionTrigger className="text-left">
                          {"Source "}
                          {idx + 1}
                          {": "}
                          {s.source}
                        </AccordionTrigger>
                        <AccordionContent className="text-white/80">
                          <div className="text-sm">
                            <div className="mb-1">
                              <Badge variant="secondary" className="text-black">
                                Similarity: {s.similarity.toFixed(3)}
                              </Badge>
                            </div>
                            <div className="whitespace-pre-wrap">{s.preview || "No preview available."}</div>
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
