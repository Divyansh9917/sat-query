'use client'

import { useState } from 'react'
import { Send, Sparkles } from 'lucide-react'

const SUGGESTIONS = [
  'Highlight water bodies',
  'Count buildings in this zone',
  'What has changed between these images?',
  'Detect roads and infrastructure',
  'Identify deforestation areas',
]

interface Props {
  query: string
  onChange: (v: string) => void
  onSubmit: () => void
  loading: boolean
  disabled: boolean   // disables the submit button only
}

export default function QueryBar({ query, onChange, onSubmit, loading, disabled }: Props) {
  const [focused, setFocused] = useState(false)

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !disabled && !loading) onSubmit()
  }

  return (
    <section className="bg-[#0A0A0A] border border-[#262626] p-5 flex flex-col gap-4">
      <h2 className="font-display font-semibold text-xs tracking-widest text-[#9CA3AF] uppercase">
        Natural Language Query
      </h2>

      {/* Input row */}
      <div
        className={`flex items-center gap-3 border px-4 py-3 transition-all duration-200 bg-[#050505] ${
          focused ? 'border-[#F96515]' : 'border-[#262626]'
        }`}
      >
        <Sparkles size={16} className="text-[#9CA3AF] shrink-0" />
        <input
          id="query-input"
          type="text"
          value={query}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask anything about the image…"
          disabled={loading}
          className="flex-1 bg-transparent text-sm text-[#F3F4F6] placeholder:text-[#9CA3AF] outline-none disabled:opacity-50"
        />
        <button
          id="analyze-btn"
          onClick={onSubmit}
          disabled={disabled || loading || !query.trim()}
          className="flex items-center gap-2 text-xs font-bold tracking-widest uppercase px-5 py-2 border border-[#F96515] text-[#F96515] hover:bg-[#F96515]/10 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <span className="h-3.5 w-3.5 rounded-full border-2 border-[#F96515]/30 border-t-[#F96515] animate-spin" />
              Analyzing…
            </>
          ) : (
            <>
              <Send size={13} />
              Analyze
            </>
          )}
        </button>
      </div>

      {/* Quick suggestion chips */}
      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            id={`suggestion-${s.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')}`}
            onClick={() => { onChange(s); }}
            disabled={disabled || loading}
            className={`text-xs px-3 py-1.5 border transition-all tracking-wide
              ${query === s
                ? 'border-[#F96515] bg-[#F96515]/10 text-[#F96515]'
                : 'border-[#262626] bg-transparent text-[#9CA3AF] hover:border-[#F96515] hover:text-[#F96515] hover:bg-[#F96515]/10'
              } disabled:opacity-40`}
          >
            {s}
          </button>
        ))}
      </div>
    </section>
  )
}
