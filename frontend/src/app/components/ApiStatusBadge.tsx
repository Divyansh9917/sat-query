'use client'

import { useEffect, useState } from 'react'
import { Wifi, WifiOff, Loader2 } from 'lucide-react'
import type { ApiStatus } from '../types'

const API_BASE = 'http://localhost:8000'

interface Props {
  onStatusChange?: (status: ApiStatus) => void
}

export default function ApiStatusBadge({ onStatusChange }: Props) {
  const [status, setStatus] = useState<ApiStatus>('checking')

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) })
        const data = await res.json()
        const next: ApiStatus = data.status === 'online' ? 'online' : 'offline'
        setStatus(next)
        onStatusChange?.(next)
      } catch {
        setStatus('offline')
        onStatusChange?.('offline')
      }
    }

    check()
    const id = setInterval(check, 10_000)
    return () => clearInterval(id)
  }, [onStatusChange])

  const cfg = {
    online:   { Icon: Wifi,    label: 'API Online',   dot: 'bg-[#F96515]',  border: 'border-[#F96515]/40', text: 'text-[#F96515]', bg: 'bg-[#F96515]/10' },
    offline:  { Icon: WifiOff, label: 'API Offline',  dot: 'bg-red-500',    border: 'border-red-500/40',   text: 'text-red-400',   bg: 'bg-red-500/10'   },
    checking: { Icon: Loader2, label: 'Connecting…',  dot: 'bg-[#9CA3AF]',  border: 'border-[#262626]',    text: 'text-[#9CA3AF]', bg: 'bg-transparent'  },
  }[status]

  return (
    <span
      id="api-status-badge"
      className={`flex items-center gap-2 text-xs font-bold tracking-widest uppercase px-3 py-1.5 border ${cfg.border} ${cfg.text} ${cfg.bg}`}
    >
      <span className="relative flex h-2 w-2">
        {status === 'online' && (
          <span className={`animate-pulse-ring absolute inline-flex h-full w-full ${cfg.dot} opacity-75`} />
        )}
        <span className={`relative inline-flex h-2 w-2 ${cfg.dot}`} />
      </span>
      <cfg.Icon
        size={12}
        className={status === 'checking' ? 'animate-spin' : ''}
      />
      {cfg.label}
    </span>
  )
}
