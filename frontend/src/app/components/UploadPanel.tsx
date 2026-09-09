'use client'

import { useCallback, useState } from 'react'
import { Upload, ImageIcon, X, Layers, GitCompare, Radio } from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

type AnalysisMode = 'single' | 'bitemporal' | 'optical-sar'

interface Props {
  onFileSelect: (file: File | null) => void
  onFileBSelect: (file: File | null) => void
  useSample: boolean
  onToggleSample: (v: boolean) => void
  fileA: File | null
  fileB: File | null
}

// ── Constants ─────────────────────────────────────────────────────────────────

const ACCEPTED = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']

const MODES: { id: AnalysisMode; label: string; Icon: React.ElementType; description: string }[] = [
  {
    id: 'single',
    label: 'Single Image',
    Icon: Layers,
    description: 'Grounding, segmentation, or VQA on one scene',
  },
  {
    id: 'bitemporal',
    label: 'Bi-temporal',
    Icon: GitCompare,
    description: 'Change detection across two timestamps',
  },
  {
    id: 'optical-sar',
    label: 'Optical-SAR',
    Icon: Radio,
    description: 'Cross-modal fusion of optical + radar data',
  },
]

const ZONE_CONFIG: Record<AnalysisMode, { slotA: string; slotB?: string }> = {
  single:       { slotA: 'Target Satellite Image' },
  bitemporal:   { slotA: 'Pre-event Image (T1)',  slotB: 'Post-event Image (T2)'  },
  'optical-sar': { slotA: 'Optical Image (RGB)',  slotB: 'SAR Image (Radar)'       },
}

// ── FileSlot ──────────────────────────────────────────────────────────────────

function FileSlot({
  label,
  sublabel,
  file,
  onSelect,
  onClear,
  id,
}: {
  label: string
  sublabel?: string
  file: File | null
  onSelect: (f: File) => void
  onClear: () => void
  id: string
}) {
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      const f = e.dataTransfer.files[0]
      if (f) onSelect(f)
    },
    [onSelect]
  )

  return (
    <div
      id={id}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => {
        if (!file) {
          const inp = document.getElementById(`file-input-${id}`) as HTMLInputElement
          inp?.click()
        }
      }}
      className={`group relative flex flex-col items-center justify-center gap-3 border-2 border-dashed p-6 transition-all duration-200 cursor-pointer min-h-[160px]
        ${dragging
          ? 'border-[#F96515] bg-[#F96515]/10'
          : file
          ? 'border-[#F96515]/60 bg-[#F96515]/5'
          : 'border-[#262626] bg-[#050505] hover:border-[#F96515] hover:bg-[#F96515]/5'
        }`}
    >
      <input
        id={`file-input-${id}`}
        type="file"
        accept={ACCEPTED.join(',')}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onSelect(f)
          e.target.value = ''
        }}
      />

      {file ? (
        <>
          {/* Clear button */}
          <button
            onClick={(e) => { e.stopPropagation(); onClear() }}
            className="absolute top-2.5 right-2.5 p-1 text-[#9CA3AF] hover:text-white transition-colors"
          >
            <X size={13} />
          </button>

          {/* Slot label */}
          <div className="absolute top-2.5 left-3">
            <span className="text-[10px] font-bold tracking-widest uppercase text-[#F96515]">
              {label}
            </span>
          </div>

          <ImageIcon size={20} className="text-[#F96515] mt-4" />
          <div className="text-center">
            <p className="text-sm font-medium text-[#F3F4F6] truncate max-w-[180px]">{file.name}</p>
            <p className="text-xs text-[#9CA3AF] mt-0.5">{(file.size / 1024).toFixed(0)} KB</p>
          </div>
        </>
      ) : (
        <>
          {/* Slot label */}
          <div className="absolute top-2.5 left-3">
            <span className="text-[10px] font-bold tracking-widest uppercase text-[#9CA3AF] group-hover:text-[#F96515] transition-colors">
              {label}
            </span>
          </div>

          <div className="p-3 border border-[#262626] bg-[#0A0A0A] group-hover:border-[#F96515]/40 transition-colors mt-4">
            <Upload size={18} className="text-[#9CA3AF] group-hover:text-[#F96515] transition-colors" />
          </div>
          <div className="text-center">
            <p className="text-xs text-[#9CA3AF] group-hover:text-[#F3F4F6] transition-colors">
              Drag & drop or click
            </p>
            {sublabel && (
              <p className="text-[10px] text-[#6B7280] mt-0.5">{sublabel}</p>
            )}
            <p className="text-[10px] text-[#6B7280] mt-0.5">{ACCEPTED.join(', ')}</p>
          </div>
        </>
      )}
    </div>
  )
}

// ── UploadPanel ───────────────────────────────────────────────────────────────

export default function UploadPanel({
  onFileSelect,
  onFileBSelect,
  useSample,
  onToggleSample,
  fileA,
  fileB,
}: Props) {
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('single')

  const zoneConfig = ZONE_CONFIG[analysisMode]
  const isTwoSlot  = !!zoneConfig.slotB

  return (
    <section className="bg-[#0A0A0A] border border-[#262626] flex flex-col">

      {/* ── Mode segmented toggle ───────────────────────────────────────────── */}
      <div className="flex border-b border-[#262626]">
        {MODES.map(({ id, label, Icon }) => {
          const active = analysisMode === id
          return (
            <button
              key={id}
              id={`mode-tab-${id}`}
              onClick={() => {
                setAnalysisMode(id)
                // Clear file B when switching to single-slot mode
                if (id === 'single') onFileBSelect(null)
              }}
              className={`flex-1 flex flex-col items-center gap-1 px-3 py-3 text-[10px] font-bold tracking-widest uppercase transition-all border-b-2 -mb-px
                ${active
                  ? 'border-[#F96515] text-[#F96515] bg-[#F96515]/10'
                  : 'border-transparent text-[#9CA3AF] hover:text-[#F3F4F6] hover:bg-[#262626]/40'
                }`}
            >
              <Icon size={14} />
              <span className="hidden sm:block">{label}</span>
            </button>
          )
        })}
      </div>

      {/* ── Mode description strip ──────────────────────────────────────────── */}
      <div className="px-5 py-2.5 border-b border-[#262626] flex items-center justify-between gap-3">
        <p className="text-[11px] text-[#9CA3AF] tracking-wide">
          {MODES.find(m => m.id === analysisMode)?.description}
        </p>

        {/* Sample toggle */}
        <button
          id="sample-toggle"
          onClick={() => onToggleSample(!useSample)}
          className={`shrink-0 flex items-center gap-1.5 text-[10px] font-bold tracking-widest uppercase px-3 py-1.5 border transition-all ${
            useSample
              ? 'border-[#F96515] text-[#F96515] bg-[#F96515]/10'
              : 'border-[#262626] text-[#9CA3AF] hover:border-[#F96515] hover:text-[#F96515] hover:bg-[#F96515]/10'
          }`}
        >
          <ImageIcon size={11} />
          Sample
        </button>
      </div>

      {/* ── Upload zones ────────────────────────────────────────────────────── */}
      <div className="p-5">
        {useSample ? (
          <div className="flex items-center gap-3 p-4 bg-[#F96515]/5 border border-[#F96515]/30">
            <ImageIcon size={18} className="text-[#F96515] shrink-0" />
            <div>
              <p className="text-sm font-semibold text-[#F96515]">Sample Image Active</p>
              <p className="text-xs text-[#9CA3AF] mt-0.5">
                Preloaded Sentinel-2 optical image · river delta region
              </p>
            </div>
          </div>
        ) : (
          <div className={`grid gap-3 ${isTwoSlot ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
            <FileSlot
              id="file-a"
              label={zoneConfig.slotA}
              sublabel={
                analysisMode === 'bitemporal'   ? 'Earlier acquisition date' :
                analysisMode === 'optical-sar'  ? 'GeoTIFF / PNG / JPEG'    : undefined
              }
              file={fileA}
              onSelect={onFileSelect}
              onClear={() => onFileSelect(null)}
            />
            {isTwoSlot && (
              <FileSlot
                id="file-b"
                label={zoneConfig.slotB!}
                sublabel={
                  analysisMode === 'bitemporal'  ? 'Later acquisition date'  :
                  analysisMode === 'optical-sar' ? 'SAR intensity / complex'  : undefined
                }
                file={fileB}
                onSelect={onFileBSelect}
                onClear={() => onFileBSelect(null)}
              />
            )}
          </div>
        )}
      </div>

      {/* ── Footer: accepted formats ─────────────────────────────────────────── */}
      <div className="px-5 pb-4 flex items-center gap-2">
        <span className="text-[10px] font-bold tracking-widest uppercase text-[#6B7280]">Accepted:</span>
        {ACCEPTED.map(ext => (
          <span key={ext} className="text-[10px] px-1.5 py-0.5 border border-[#262626] text-[#9CA3AF] font-mono">
            {ext}
          </span>
        ))}
      </div>
    </section>
  )
}
