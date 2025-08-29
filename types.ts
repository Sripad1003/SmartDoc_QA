export type HealthResponse = {
  status?: string
  documents_processed?: number
  chunks_indexed?: number
  [key: string]: any
}

export type DocumentListResponse = {
  documents: {
    doc_id: string
    filename: string
    chunk_count: number
    text_length: number
    processed_at: string
  }[]
}

export type UploadResponse = {
  summary?: {
    total_files: number
    successful: number
    failed: number
    errors?: { filename: string; error: string }[]
  }
  details?: {
    filename: string
    doc_id?: string
    status: "processed" | "failed"
    error?: string
  }[]
  [k: string]: any
}

export type SourceItem = {
  source: string
  similarity: number
  preview?: string
}

export type AskResponse = {
  session_id: string
  answer: string
  confidence: number
  response_time: number
  sources?: SourceItem[]
  [k: string]: any
}

export type GenerateQuestionsResponse = {
  questions_generated: number
  filename?: string
  questions: string[]
  questions_data?: {
    question: string
    question_type?: string
  }[]
  [k: string]: any
}

export type PredictionItem = {
  question: string
  expected?: string
  predicted?: string
  f1_score?: number
  semantic_score?: number
  contains_answer?: boolean
  response_time?: number
  question_type?: string
  [k: string]: any
}

export type EvaluateResponse = {
  results: {
    average_f1?: number
    average_semantic?: number
    accuracy_rate?: number
    total_questions?: number
    average_response_time?: number
    question_types?: Record<string, number>
    predictions?: PredictionItem[]
    f1_scores?: number[]
    semantic_scores?: number[]
    [k: string]: any
  }
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
