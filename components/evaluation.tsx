"use client"

import { useEffect, useMemo, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Separator } from "@/components/ui/separator"
import { Brain, ListChecks, LineChart } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { evaluateDocument, generateQuestions, getDocuments } from "@/lib/api"
import type { DocumentListResponse, GenerateQuestionsResponse, EvaluateResponse, PredictionItem } from "@/types"
import { ScoresBarCharts } from "@/components/scores-charts"

export default function Evaluation() {
  const { toast } = useToast()
  const [documents, setDocuments] = useState<DocumentListResponse["documents"]>([])
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null)

  const [generating, setGenerating] = useState(false)
  const [questionsData, setQuestionsData] = useState<GenerateQuestionsResponse | null>(null)

  const [runningEval, setRunningEval] = useState(false)
  const [evalData, setEvalData] = useState<EvaluateResponse["results"] | null>(null)

  useEffect(() => {
    ;(async () => {
      try {
        const resp = await getDocuments()
        setDocuments(resp.documents)
        if (resp.documents.length > 0) {
          setSelectedDocId(resp.documents[0].doc_id)
        }
      } catch {
        toast({ title: "Failed to load documents", variant: "destructive" })
      }
    })()
  }, [toast])

  const selectedDoc = useMemo(
    () => documents.find((d) => d.doc_id === selectedDocId) || null,
    [documents, selectedDocId],
  )

  async function onGenerateQuestions() {
    if (!selectedDocId) return
    setGenerating(true)
    try {
      const seed = Date.now()
      const resp = await generateQuestions(selectedDocId, 8, seed)
      setQuestionsData(resp)
      toast({ title: "Questions generated", description: `${resp.questions_generated} questions ready.` })
    } catch (e: any) {
      toast({ title: "Generation failed", description: e?.message || "Unknown error", variant: "destructive" })
    } finally {
      setGenerating(false)
    }
  }

  async function onRunEvaluation() {
    if (!selectedDocId) return
    setRunningEval(true)
    try {
      const resp = await evaluateDocument(selectedDocId)
      setEvalData(resp.results)
      toast({ title: "Evaluation complete" })
    } catch (e: any) {
      toast({ title: "Evaluation failed", description: e?.message || "Unknown error", variant: "destructive" })
    } finally {
      setRunningEval(false)
    }
  }

  function gradeFromSemantic(score: number) {
    if (score >= 0.9) return { label: "A+ (Excellent)", color: "bg-emerald-600" }
    if (score >= 0.8) return { label: "A (Very Good)", color: "bg-green-600" }
    if (score >= 0.7) return { label: "B (Good)", color: "bg-yellow-600" }
    if (score >= 0.6) return { label: "C (Acceptable)", color: "bg-orange-600" }
    return { label: "D (Needs Improvement)", color: "bg-red-600" }
  }

  const questionTypesGrouped = useMemo(() => {
    if (!questionsData?.questions_data) return null
    const map = new Map<string, { question: string }[]>()
    for (const q of questionsData.questions_data) {
      const key = (q.question_type || "general") as string
      if (!map.has(key)) map.set(key, [])
      map.get(key)!.push({ question: q.question })
    }
    return Array.from(map.entries())
  }, [questionsData])

  return (
    <div className="flex flex-col gap-8">
      <Card className="bg-white/5 text-white border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            AI-Powered Evaluation
          </CardTitle>
          <CardDescription className="text-white/70">
            Gemini AI generates intelligent questions to evaluate retrieval and answer quality.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <div className="md:col-span-2">
              <label className="text-sm text-white/80">Select document</label>
              <Select value={selectedDocId || ""} onValueChange={setSelectedDocId}>
                <SelectTrigger className="bg-black/30 border-white/10 text-white">
                  <SelectValue placeholder="Choose a document" />
                </SelectTrigger>
                <SelectContent>
                  {documents.map((d) => (
                    <SelectItem key={d.doc_id} value={d.doc_id}>
                      {d.filename} ({d.chunk_count} chunks)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end gap-2">
              <Button
                variant="outline"
                onClick={onGenerateQuestions}
                disabled={!selectedDocId || generating}
                className="border-white/20 text-white bg-transparent"
              >
                <ListChecks className="mr-2 h-4 w-4" />
                {generating ? "Generating..." : "Generate AI Questions"}
              </Button>
              <Button onClick={onRunEvaluation} disabled={!selectedDocId || runningEval}>
                <LineChart className="mr-2 h-4 w-4" />
                {runningEval ? "Evaluating..." : "Run Full Evaluation"}
              </Button>
            </div>
          </div>

          {selectedDoc && (
            <div className="grid grid-cols-3 gap-3">
              <Card className="bg-white/5 border-white/10">
                <CardContent className="p-4">
                  <div className="text-xs text-white/60">Chunks</div>
                  <div className="text-2xl font-semibold">{selectedDoc.chunk_count}</div>
                </CardContent>
              </Card>
              <Card className="bg-white/5 border-white/10">
                <CardContent className="p-4">
                  <div className="text-xs text-white/60">Text Length</div>
                  <div className="text-2xl font-semibold">{selectedDoc.text_length.toLocaleString()}</div>
                </CardContent>
              </Card>
              <Card className="bg-white/5 border-white/10">
                <CardContent className="p-4">
                  <div className="text-xs text-white/60">Est. Questions</div>
                  <div className="text-2xl font-semibold">
                    {Math.min(8, Math.max(5, Math.floor(selectedDoc.chunk_count / 3)))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Generated Questions */}
      {questionsData && (
        <Card className="bg-white/5 text-white border-white/10">
          <CardHeader>
            <CardTitle>AI-Generated Questions</CardTitle>
            <CardDescription className="text-white/70">
              {`Generated ${questionsData.questions_generated} questions for ${questionsData.filename}`}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {questionTypesGrouped && questionTypesGrouped.length > 0 ? (
              questionTypesGrouped.map(([qType, items]) => (
                <div key={qType} className="space-y-2">
                  <div className="font-semibold">
                    {qType
                      .replace(/_/g, " ")
                      .replace(/^gemini_/, "")
                      .toUpperCase()}
                  </div>
                  <ul className="list-disc ml-5 text-white/90">
                    {items.map((it, idx) => (
                      <li key={`${qType}-${idx}`}>{it.question}</li>
                    ))}
                  </ul>
                  <Separator className="bg-white/10" />
                </div>
              ))
            ) : (
              <ul className="list-disc ml-5 text-white/90">
                {questionsData.questions.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            )}
            <div className="text-sm text-white/60">Powered by: Google Gemini AI</div>
          </CardContent>
        </Card>
      )}

      {/* Evaluation Results */}
      {evalData && (
        <Card className="bg-white/5 text-white border-white/10">
          <CardHeader>
            <CardTitle>Performance Summary</CardTitle>
            <CardDescription className="text-white/70">Evaluation metrics and detailed analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <Metric label="F1 Score" value={evalData.average_f1?.toFixed(3) || "0.000"} />
              <Metric label="Semantic Score" value={evalData.average_semantic?.toFixed(3) || "0.000"} />
              <Metric label="Accuracy Rate" value={`${((evalData.accuracy_rate || 0) * 100).toFixed(1)}%`} />
              <Metric label="Questions" value={`${evalData.total_questions || 0}`} />
              <Metric label="Avg Time" value={`${(evalData.average_response_time || 0).toFixed(2)}s`} />
            </div>

            {/* Question types */}
            {evalData.question_types && Object.keys(evalData.question_types).length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-2">AI Question Types Generated</h3>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(evalData.question_types).map(([k, v]) => (
                    <Badge key={k} variant="secondary" className="text-black">
                      {k.replace(/^gemini_/, "").replace(/_/g, " ")}: {v as number}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Grade */}
            <div>
              <h3 className="text-lg font-semibold mb-2">Performance Assessment</h3>
              <GradeBadge semantic={evalData.average_semantic || 0} />
            </div>

            {/* Detailed predictions */}
            {evalData.predictions && evalData.predictions.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-2">Detailed Question Analysis</h3>
                <Accordion type="single" collapsible className="w-full">
                  {evalData.predictions.map((p: PredictionItem, idx: number) => {
                    const qt = (p.question_type || "general").replace(/^gemini_/, "").replace(/_/g, " ")
                    return (
                      <AccordionItem key={idx} value={`pred-${idx}`} className="border-white/10">
                        <AccordionTrigger className="text-left">
                          {`Q${idx + 1} [${qt}]: ${p.question.slice(0, 60)}...`}
                        </AccordionTrigger>
                        <AccordionContent className="text-sm text-white/90 space-y-2">
                          <div>
                            <span className="font-semibold">Question:</span> {p.question}
                          </div>
                          <div>
                            <span className="font-semibold">Expected:</span> {p.expected?.slice(0, 300)}...
                          </div>
                          <div>
                            <span className="font-semibold">Predicted:</span> {p.predicted?.slice(0, 300)}...
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">
                            <MetricCompact label="F1" value={p.f1_score?.toFixed(3) || "0.000"} />
                            <MetricCompact label="Semantic" value={p.semantic_score?.toFixed(3) || "0.000"} />
                            <MetricCompact label="Contains" value={p.contains_answer ? "Yes" : "No"} />
                            <MetricCompact label="Time" value={`${(p.response_time || 0).toFixed(2)}s`} />
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    )
                  })}
                </Accordion>
              </div>
            )}

            {/* Distributions */}
            {evalData.f1_scores &&
              evalData.semantic_scores &&
              evalData.f1_scores.length > 0 &&
              evalData.semantic_scores.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold">Performance Distributions</h3>
                  <ScoresBarCharts f1={evalData.f1_scores} semantic={evalData.semantic_scores} />
                </div>
              )}

            {/* Insights */}
            <div className="text-sm text-white/80">
              <p>
                <strong>AI Evaluation Insights:</strong>
              </p>
              <ul className="list-disc ml-5">
                <li>{`Generated ${evalData.total_questions || 0} diverse questions across multiple categories`}</li>
                <li>Includes factual, analytical, and conceptual queries</li>
                <li>{`Average response quality grade: ${gradeFromSemantic(evalData.average_semantic || 0).label}`}</li>
                <li>{`Accuracy rate: ${((evalData.accuracy_rate || 0) * 100).toFixed(0)}%`}</li>
                <li>{`Average response time: ${(evalData.average_response_time || 0).toFixed(2)}s`}</li>
              </ul>
            </div>
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
function MetricCompact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-white/60">{label}</div>
      <div className="text-base font-medium">{value}</div>
    </div>
  )
}
function GradeBadge({ semantic }: { semantic: number }) {
  const grade =
    semantic >= 0.9
      ? "A+ (Excellent)"
      : semantic >= 0.8
        ? "A (Very Good)"
        : semantic >= 0.7
          ? "B (Good)"
          : semantic >= 0.6
            ? "C (Acceptable)"
            : "D (Needs Improvement)"
  const color =
    semantic >= 0.9
      ? "bg-emerald-600"
      : semantic >= 0.8
        ? "bg-green-600"
        : semantic >= 0.7
          ? "bg-yellow-600"
          : semantic >= 0.6
            ? "bg-orange-600"
            : "bg-red-600"
  return <Badge className={`${color} text-white text-base py-1 px-3`}>{`Overall Grade (Semantic): ${grade}`}</Badge>
}
