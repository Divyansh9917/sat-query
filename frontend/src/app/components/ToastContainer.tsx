'use client'

import { useEffect } from 'react'
import { CheckCircle2, XCircle, Info, X } from 'lucide-react'
import type { Toast } from '../types'

interface Props {
  toasts: Toast[]
  onRemove: (id: string) => void
}

const cfg = {
  success: { Icon: CheckCircle2, color: '#F96515', border: 'border-[#F96515]/40', bg: 'bg-[#F96515]/10' },
  error:   { Icon: XCircle,      color: '#ef4444', border: 'border-red-500/40',   bg: 'bg-red-500/10'   },
  info:    { Icon: Info,         color: '#9CA3AF', border: 'border-[#262626]',     bg: 'bg-[#0A0A0A]'   },
}

function ToastItem({ toast, onRemove }: { toast: Toast; onRemove: () => void }) {
  const { Icon, color, border, bg } = cfg[toast.type]

  useEffect(() => {
    const t = setTimeout(onRemove, 4500)
    return () => clearTimeout(t)
  }, [onRemove])

  return (
    <div
      id={`toast-${toast.id}`}
      className={`flex items-start gap-3 px-4 py-3 border ${border} ${bg} shadow-xl animate-fade-up`}
      style={{ minWidth: 260, maxWidth: 380 }}
    >
      <Icon size={16} style={{ color, marginTop: 1, flexShrink: 0 }} />
      <p className="text-sm text-[#F3F4F6] flex-1 leading-snug">{toast.message}</p>
      <button
        onClick={onRemove}
        className="text-[#9CA3AF] hover:text-white transition-colors mt-0.5"
      >
        <X size={14} />
      </button>
    </div>
  )
}

export default function ToastContainer({ toasts, onRemove }: Props) {
  if (!toasts.length) return null

  return (
    <div className="toast flex flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onRemove={() => onRemove(t.id)} />
      ))}
    </div>
  )
}
