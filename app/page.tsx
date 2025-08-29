"use client"

import { useEffect, useMemo, useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Activity, BookCopy, Brain, History, RefreshCw } from "lucide-react"
import UploadAndQA from "@/components/upload-and-qa"
import Evaluation from "@/components/evaluation"
import ConversationHistory from "@/components/history"
import { useToast } from "@/hooks/use-toast"
import { Toaster } from "@/components/ui/toaster"
import { getHealth } from "@/lib/api"
import type { HealthResponse } from "@/types"

export default function HomePage() {
  const { toast } = useToast()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  const statusBadge = useMemo(() => {
    if (!health) return <Badge variant="secondary">Unknown</Badge>
    return <Badge className="bg-emerald-600 hover:bg-emerald-700 text-white">Healthy</Badge>
  }, [health])

  async function refreshHealth() {
    try {
      setLoading(true)
      const h = await getHealth()
      setHealth(h)
      setLastChecked(new Date())
    } catch (e: any) {
      toast({
        title: "Backend not reachable",
        description: "Make sure FastAPI is running at your configured API base URL.",
        variant: "destructive",
      })
      setHealth(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refreshHealth()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white">
      <section className="border-b border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.06)_0%,rgba(255,255,255,0)_100%)]">
        <div className="mx-auto max-w-7xl px-4 py-8">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">Simple Document Q&A System</h1>
              <p className="text-sm text-white/70 mt-1">
                AI-powered document upload, Q&amp;A, and evaluation with the existing FastAPI backend.
              </p>
            </div>
            <div className="flex items-center gap-2">
              {statusBadge}
              <Button
                size="sm"
                variant="secondary"
                onClick={refreshHealth}
                disabled={loading}
                className="bg-white/10 hover:bg-white/20 text-white border border-white/10"
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                {loading ? "Checking..." : "Check backend"}
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 mt-6 md:grid-cols-3">
            <Card className="bg-white/5 border-white/10 text-white">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <BookCopy className="h-4 w-4" />
                  Documents
                </CardTitle>
                <CardDescription className="text-white/60">Processed documents</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold tabular-nums">{health?.documents_processed ?? 0}</div>
              </CardContent>
            </Card>

            <Card className="bg-white/5 border-white/10 text-white">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Brain className="h-4 w-4" />
                  Chunks
                </CardTitle>
                <CardDescription className="text-white/60">Indexed knowledge chunks</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold tabular-nums">{health?.chunks_indexed ?? 0}</div>
              </CardContent>
            </Card>

            <Card className="bg-white/5 border-white/10 text-white">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  Last Checked
                </CardTitle>
                <CardDescription className="text-white/60">Backend health status</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-xl font-medium">{lastChecked ? lastChecked.toLocaleTimeString() : "—"}</div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-8">
        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-white/10 text-white">
            <TabsTrigger value="upload" className="data-[state=active]:bg-white data-[state=active]:text-black">
              Upload &amp; Q&amp;A
            </TabsTrigger>
            <TabsTrigger value="evaluation" className="data-[state=active]:bg-white data-[state=active]:text-black">
              Evaluation
            </TabsTrigger>
            <TabsTrigger value="history" className="data-[state=active]:bg-white data-[state=active]:text-black">
              History
            </TabsTrigger>
          </TabsList>

          <Separator className="my-4 bg-white/10" />

          <TabsContent value="upload">
            <UploadAndQA />
          </TabsContent>

          <TabsContent value="evaluation">
            <Evaluation />
          </TabsContent>

          <TabsContent value="history">
            <ConversationHistory />
          </TabsContent>
        </Tabs>
      </section>

      <footer className="border-t border-white/10 mt-8">
        <div className="mx-auto max-w-7xl px-4 py-6 text-sm text-white/60 flex items-center gap-2">
          <History className="h-4 w-4" />
          <span>{"Frontend: Next.js (App Router) • Backend: FastAPI • Q&A System"}</span>
        </div>
      </footer>

      <Toaster />
    </main>
  )
}
