export type HealthResponse = {
  status: string
  timestamp: string
  documents_processed: number
  chunks_indexed: number
}

export type DocumentListResponse = {
  total_documents: number
  documents: {
    doc_id: string
    filename: string
    processed_at: string
    chunk_count: number
    text_length: number
  }[]
}

export type UploadResponse = {
  documents: {
    filename: string
    doc_id?: string
    status: "processed" | "failed"
    chunks?: number
    text_length?: number
    processing_time?: number
    error?: string
  }[]
  summary: {
    total_files: number
    successful: number
    failed: number
  }
  message: string
}

export type SourceItem = {
  source: string
  similarity: number
  preview?: string
}

export type AskResponse = {
  answer: string
  sources: SourceItem[]
  confidence: number
  session_id: string
  response_time: number
}

export type GenerateQuestionsResponse = {
  doc_id: string
  filename: string
  questions_generated: number
  questions: string[]
  questions_data?: {
    question: string
    question_type?: string
  }[]
}

export type PredictionItem = {
  question: string
  expected: string
  predicted: string
  f1_score: number
  semantic_score: number
  contains_answer: boolean
  response_time: number
  confidence: number
  question_type?: string
}

export type EvaluateResponse = {
  message: string
  results: {
    average_f1: number
    average_semantic: number
    accuracy_rate: number
    total_questions: number
    average_response_time: number
    question_types?: Record<string, number>
    predictions: PredictionItem[]
    f1_scores?: number[]
    semantic_scores?: number[]
  }
}
