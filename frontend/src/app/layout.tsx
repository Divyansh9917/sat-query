import type { Metadata } from 'next'
import { Inter, Space_Grotesk, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'SatQuery AI | Agentic Multimodal RS Assistant',
  description:
    'Agentic Multimodal Remote Sensing AI — query satellite imagery with natural language, powered by GroundingDINO, CLIP, and GeoSAM.',
  keywords: ['satellite', 'remote sensing', 'AI', 'geospatial', 'multimodal'],
  openGraph: {
    title: 'SatQuery AI',
    description: 'Agentic Multimodal Remote Sensing Analysis',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  )
}
