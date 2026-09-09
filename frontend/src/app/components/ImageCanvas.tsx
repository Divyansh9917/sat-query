'use client'

import { useEffect, useRef } from 'react'
import { ScanLine, Crosshair } from 'lucide-react'
import type { AnalysisResult } from '../types'

interface Props {
  imageSrc: string | null
  result: AnalysisResult | null
  loading: boolean
}

const LABEL_COLORS: Record<string, string> = {
  'water body': '#F96515',
  building:     '#f59e0b',
  road:         '#a855f7',
  forest:       '#10b981',
  default:      '#F96515',
}

function getColor(label: string) {
  return LABEL_COLORS[label] ?? LABEL_COLORS.default
}

export default function ImageCanvas({ imageSrc, result, loading }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !imageSrc) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = imageSrc

    img.onload = () => {
      imgRef.current = img
      canvas.width  = img.naturalWidth
      canvas.height = img.naturalHeight

      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0)

      if (result?.detections?.length) {
        result.detections.forEach(({ label, confidence, box }) => {
          const [x1, y1, x2, y2] = box
          const color = getColor(label)

          // Translucent fill
          ctx.fillStyle = `${color}22`
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1)

          // Border
          ctx.strokeStyle = color
          ctx.lineWidth   = 2
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

          // Corner accents
          const cs = 12
          ctx.lineWidth = 3
          const corners = [
            { cx: x1, cy: y1, dx: cs,  dy: cs  },
            { cx: x2, cy: y1, dx: -cs, dy: cs  },
            { cx: x1, cy: y2, dx: cs,  dy: -cs },
            { cx: x2, cy: y2, dx: -cs, dy: -cs },
          ]
          corners.forEach(({ cx, cy, dx, dy }) => {
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + dx, cy); ctx.stroke()
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + dy); ctx.stroke()
          })

          // Label pill
          const pct = Math.round(confidence * 100)
          const text = `${label}  ${pct}%`
          ctx.font = 'bold 11px Inter, sans-serif'
          const tw = ctx.measureText(text).width
          const ph = 18
          const px = x1
          const py = y1 - ph - 4

          ctx.fillStyle = color
          ctx.beginPath()
          ;(ctx as CanvasRenderingContext2D & { roundRect: (...a: number[]) => void })
            .roundRect(px, py, tw + 16, ph, 2)
          ctx.fill()

          ctx.fillStyle = '#050505'
          ctx.fillText(text, px + 8, py + ph - 5)
        })
      }
    }
  }, [imageSrc, result])

  return (
    <section className="bg-[#0A0A0A] border border-[#262626] p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="font-display font-semibold text-xs tracking-widest text-[#9CA3AF] uppercase">
          Analysis Canvas
        </h2>
        {result && (
          <span
            id="detection-badge"
            className="flex items-center gap-1.5 text-xs font-bold tracking-widest uppercase px-3 py-1 border border-[#F96515]/40 text-[#F96515] bg-[#F96515]/10"
          >
            <Crosshair size={11} />
            {result.detections.length} detection{result.detections.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      <div
        id="canvas-container"
        className="relative w-full overflow-hidden bg-[#050505] border border-[#262626] min-h-[280px] flex items-center justify-center"
      >
        {loading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-[#050505]/90 backdrop-blur-sm">
            {/* Scan line — orange */}
            <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-[#F96515] to-transparent animate-scan opacity-80" />
            <div className="flex flex-col items-center gap-3">
              <div className="relative h-10 w-10">
                <div className="absolute inset-0 border border-[#F96515]/20 animate-ping" />
                <div className="absolute inset-1 border-2 border-t-[#F96515] border-r-[#F96515]/40 border-b-transparent border-l-transparent animate-spin" />
              </div>
              <p className="text-xs font-mono text-[#F96515] tracking-widest uppercase">Running inference…</p>
            </div>
          </div>
        )}

        {imageSrc ? (
          <canvas
            ref={canvasRef}
            id="analysis-canvas"
            className="w-full h-auto object-contain"
            style={{ imageRendering: 'pixelated' }}
          />
        ) : (
          <div className="flex flex-col items-center gap-3 py-12 text-[#9CA3AF]">
            <ScanLine size={40} strokeWidth={1} />
            <p className="text-sm tracking-wide">Upload an image to begin analysis</p>
          </div>
        )}
      </div>
    </section>
  )
}
