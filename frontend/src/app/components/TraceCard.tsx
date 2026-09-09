'use client'

import { Clock, Cpu, Globe2, TrendingUp } from 'lucide-react'
import type { AnalysisResult } from '../types'

interface Props {
  result: AnalysisResult | null
  loading: boolean
}

function MetricRow({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: React.ElementType
  label: string
  value: React.ReactNode
  accent?: string
}) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-[#262626] last:border-0">
      <div className="flex items-center gap-2.5">
        <span className="p-1.5 border border-[#262626] bg-[#050505]">
          <Icon size={13} className="text-[#9CA3AF]" />
        </span>
        <span className="text-xs text-[#9CA3AF] tracking-wide">{label}</span>
      </div>
      <span
        className="font-mono text-xs font-medium"
        style={{ color: accent ?? '#F3F4F6' }}
      >
        {value}
      </span>
    </div>
  )
}

function ConfBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 80 ? '#F96515' :
    pct >= 60 ? '#f59e0b' : '#ef4444'

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-center">
        <span className="text-xs text-[#9CA3AF] tracking-wide uppercase">Confidence</span>
        <span className="font-mono text-xs font-bold" style={{ color }}>{pct}%</span>
      </div>
      <div className="h-1.5 bg-[#050505] border border-[#262626]">
        <div
          className="h-full transition-all duration-1000"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

export default function TraceCard({ result, loading }: Props) {
  if (!result && !loading) {
    return (
      <section className="bg-[#0A0A0A] border border-[#262626] p-5 flex flex-col gap-4">
        <h2 className="font-display font-semibold text-xs tracking-widest text-[#9CA3AF] uppercase">
          Analysis Trace
        </h2>
        <div className="flex flex-col items-center gap-2 py-8 text-[#9CA3AF]">
          <Cpu size={28} strokeWidth={1} />
          <p className="text-xs tracking-wide">Run an analysis to see the trace</p>
        </div>
      </section>
    )
  }

  if (loading) {
    return (
      <section className="bg-[#0A0A0A] border border-[#262626] p-5 flex flex-col gap-4">
        <h2 className="font-display font-semibold text-xs tracking-widest text-[#9CA3AF] uppercase">
          Analysis Trace
        </h2>
        {[1,2,3,4].map((i) => (
          <div key={i} className="h-10 animate-shimmer bg-[#050505] border border-[#262626]" />
        ))}
      </section>
    )
  }

  const { confidence, execution_time_ms, trace } = result!

  return (
    <section className="bg-[#0A0A0A] border border-[#262626] p-5 flex flex-col gap-4 animate-fade-up">
      <div className="flex items-center justify-between">
        <h2 className="font-display font-semibold text-xs tracking-widest text-[#9CA3AF] uppercase">
          Analysis Trace
        </h2>
        <span className="flex items-center gap-1.5 text-xs font-bold tracking-widest uppercase px-3 py-1 border border-[#F96515]/40 text-[#F96515] bg-[#F96515]/10">
          <TrendingUp size={11} /> Success
        </span>
      </div>

      {/* Answer */}
      <div className="p-3 bg-[#F96515]/5 border border-[#F96515]/20">
        <p className="text-sm text-[#F3F4F6] leading-relaxed">{result!.answer}</p>
      </div>

      {/* Confidence bar */}
      <ConfBar value={confidence} />

      {/* Metrics */}
      <div className="bg-[#050505] border border-[#262626] px-4">
        <MetricRow
          icon={Clock}
          label="Latency"
          value={`${execution_time_ms} ms`}
          accent="#F96515"
        />
        <MetricRow
          icon={Cpu}
          label="Model"
          value={trace.selected_model}
          accent="#F3F4F6"
        />
        <MetricRow
          icon={Globe2}
          label="CRS"
          value={trace.crs}
          accent="#9CA3AF"
        />
        <MetricRow
          icon={TrendingUp}
          label="Modality"
          value={trace.modality}
        />
      </div>

      {/* Detections summary */}
      {result!.detections.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-xs font-bold text-[#9CA3AF] uppercase tracking-widest">Detections</h3>
          {result!.detections.map((d, i) => (
            <div
              key={i}
              className="flex items-center justify-between p-2.5 bg-[#050505] border border-[#262626]"
            >
              <div className="flex items-center gap-2">
                <span
                  className="h-2 w-2"
                  style={{ background: '#F96515', boxShadow: '0 0 6px #F96515' }}
                />
                <span className="text-xs text-[#F3F4F6] capitalize tracking-wide">{d.label}</span>
              </div>
              <span className="font-mono text-xs text-[#F96515] font-bold">{Math.round(d.confidence * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
