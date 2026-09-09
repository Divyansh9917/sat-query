'use client'

import { useCallback, useState } from 'react'
import { Satellite, Layers } from 'lucide-react'
import ApiStatusBadge from './components/ApiStatusBadge'
import UploadPanel from './components/UploadPanel'
import QueryBar from './components/QueryBar'
import ImageCanvas from './components/ImageCanvas'
import TraceCard from './components/TraceCard'
import ToastContainer from './components/ToastContainer'
import HeroSection from './components/HeroSection' // 👈 1. We imported the Hero Section here
import type { AnalysisResult, ApiStatus, Toast } from './types'

const API_BASE = 'http://localhost:8000'

const SAMPLE_IMAGE = '/sample.png'

let toastIdCounter = 0
function newId() { return `t-${++toastIdCounter}` }

export default function Dashboard() {
  // ── State ─────────────────────────────────────────────────────────────────
  const [fileA, setFileA] = useState<File | null>(null)
  const [fileB, setFileB] = useState<File | null>(null)
  const [useSample, setUseSample] = useState(false)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [imageSrc, setImageSrc] = useState<string | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')

  // ── Helpers ───────────────────────────────────────────────────────────────
  const addToast = useCallback((type: Toast['type'], message: string) => {
    const id = newId()
    setToasts((prev) => [...prev, { id, type, message }])
  }, [])

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const handleFileA = useCallback((f: File | null) => {
    setFileA(f)
    setResult(null)
    if (f) {
      const url = URL.createObjectURL(f)
      setImageSrc(url)
    } else {
      setImageSrc(null)
    }
  }, [])

  const handleFileB = useCallback((f: File | null) => {
    setFileB(f)
  }, [])

  const handleToggleSample = useCallback((v: boolean) => {
    setUseSample(v)
    setResult(null)
    if (v) {
      setImageSrc(SAMPLE_IMAGE)
      setFileA(null)
      setFileB(null)
    } else {
      if (!fileA) setImageSrc(null)
    }
  }, [fileA])

  // ── Analyze ───────────────────────────────────────────────────────────────
  const handleAnalyze = useCallback(async () => {
    if (!query.trim()) {
      addToast('error', 'Please enter a query before analyzing.')
      return
    }
    if (!useSample && !fileA) {
      addToast('error', 'Please upload an image or enable the sample image.')
      return
    }
    if (apiStatus === 'offline') {
      addToast('error', 'Backend API is offline. Start it with: uvicorn api.main:app')
      return
    }

    setLoading(true)
    setResult(null)

    try {
      const form = new FormData()
      form.append('query', query)

      if (useSample) {
        const blob = await fetch(SAMPLE_IMAGE).then((r) => r.blob())
        form.append('file', blob, 'sample.png')
      } else {
        form.append('file', fileA!)
        if (fileB) form.append('file_b', fileB)
      }

      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const err = await res.text()
        throw new Error(err || `HTTP ${res.status}`)
      }

      const data: AnalysisResult = await res.json()
      setResult(data)
      addToast('success', `Analysis complete — ${data.detections.length} detection(s) found.`)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      addToast('error', `Analysis failed: ${msg}`)
    } finally {
      setLoading(false)
    }
  }, [query, useSample, fileA, fileB, apiStatus, addToast])

  // ── Render ────────────────────────────────────────────────────────────────
  const canAnalyze = (!!fileA || useSample) && !!query.trim() && !loading

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header
        id="site-header"
        className="sticky top-0 z-50 glass border-b border-[var(--border)] bg-[#050505]/80 backdrop-blur-md"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="relative p-2 rounded-xl border border-[#F96515]/30">
              <Satellite size={20} className="text-[#F96515]" />
              <span className="absolute -top-1 -right-1 h-2.5 w-2.5 rounded-full bg-emerald-400 border-2 border-[var(--bg-base)]" />
            </div>
            <div>
              <h1 className="font-display font-bold text-base sm:text-lg leading-none text-white">
                SatQuery AI
              </h1>
              <p className="text-[10px] sm:text-xs text-muted mt-0.5 leading-none">
                Agentic Multimodal RS Assistant
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-1.5 text-xs text-muted">
              <Layers size={12} />
              <span>v1.0.0</span>
            </div>
            <ApiStatusBadge onStatusChange={setApiStatus} />
          </div>
        </div>
      </header>

      {/* 👈 2. The Hero Section is dropped right here! */}
      <HeroSection />

      {/* ── Main Layout (Dashboard) ────────────────────────────────────────── */}
      <main
        id="app-dashboard"
        className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-20 grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 items-start"
      >
        {/* Left column */}
        <div className="flex flex-col gap-5">
          <UploadPanel
            fileA={fileA}
            fileB={fileB}
            useSample={useSample}
            onFileSelect={handleFileA}
            onFileBSelect={handleFileB}
            onToggleSample={handleToggleSample}
          />

          <QueryBar
            query={query}
            onChange={setQuery}
            onSubmit={handleAnalyze}
            loading={loading}
            disabled={!canAnalyze && !loading}
          />

          <ImageCanvas
            imageSrc={imageSrc}
            result={result}
            loading={loading}
          />
        </div>

        {/* Right column – trace */}
        <aside className="sticky top-[100px]">
          <TraceCard result={result} loading={loading} />
        </aside>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-gridline py-6 text-center text-xs text-muted">
        SatQuery AI · Agentic Remote Sensing · Powered by GroundingDINO + GeoSAM + CLIP
      </footer>

      {/* ── Toasts ─────────────────────────────────────────────────────────── */}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </div>
  )
}