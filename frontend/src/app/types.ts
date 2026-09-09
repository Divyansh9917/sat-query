// ── SatQuery AI – shared type definitions ─────────────────────────────────

export interface DetectionBox {
  label: string
  confidence: number
  /** [x1, y1, x2, y2] in pixels */
  box: [number, number, number, number]
}

export interface AnalysisTrace {
  selected_model: string
  modality: string
  crs: string
}

export interface AnalysisResult {
  status: 'success' | 'error'
  task: string
  query: string
  answer: string
  confidence: number
  execution_time_ms: number
  trace: AnalysisTrace
  detections: DetectionBox[]
}

export type ApiStatus = 'online' | 'offline' | 'checking'

export interface Toast {
  id: string
  type: 'success' | 'error' | 'info'
  message: string
}
