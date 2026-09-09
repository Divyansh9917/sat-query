import React from 'react';

export default function HeroSection() {
    return (
        <section className="relative min-h-screen bg-[#050505] text-white overflow-hidden px-6 md:px-24 py-12 font-sans">

            {/* BACKGROUND CIRCUIT PATTERN */}
            <div
                className="absolute inset-0 z-0 opacity-20"
                style={{
                    backgroundImage: 'linear-gradient(to right, #262626 1px, transparent 1px), linear-gradient(to bottom, #262626 1px, transparent 1px)',
                    backgroundSize: '100px 100px'
                }}
            />

            {/* NAVBAR */}
            <nav className="relative z-10 flex justify-between items-center mb-32">
                <div className="flex items-center gap-2">
                    {/* Faux Logo */}
                    <div className="w-6 h-6 border-2 border-[#F96515] rounded-sm flex items-center justify-center">
                        <div className="w-1.5 h-1.5 bg-[#F96515] rounded-full" />
                    </div>
                    <span className="font-bold text-xl tracking-tight text-white">SatQuery AI</span>
                </div>

                <div className="hidden md:flex items-center gap-8 text-xs font-semibold tracking-widest text-[#9CA3AF]">
                    <a href="#app-dashboard" className="hover:text-white transition-colors">DASHBOARD</a>
                    <a href="#architecture" className="hover:text-white transition-colors">ARCHITECTURE</a>
                    <a href="#app-dashboard" className="border border-[#F96515] text-[#F96515] px-6 py-2.5 rounded-full hover:bg-[#F96515]/10 transition-colors uppercase">
                        Run Query
                    </a>
                </div>
            </nav>

            {/* MAIN CONTENT */}
            <div className="relative z-10 max-w-4xl mt-20">
                {/* Subtitle */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-2 h-2 bg-[#F96515]" />
                    <h2 className="text-[#F96515] text-xs font-bold tracking-widest uppercase">
                        Smart India Hackathon 2026 • SIH26167
                    </h2>
                </div>

                {/* Main Title */}
                <h1 className="text-6xl md:text-8xl font-black text-[#F96515] tracking-tighter leading-[1.1] mb-8">
                    Decoding satellite data was never this easy.
                </h1>

                {/* Paragraph */}
                <p className="text-[#9CA3AF] text-lg md:text-xl max-w-2xl leading-relaxed mb-20">
                    An agentic vision-language assistant that analyzes single, cross-modal, and bi-temporal remote-sensing images through simple natural language queries. Powered by a unified RS-VLM backbone.
                </p>

                {/* STATS SECTION */}
                <div className="flex flex-col md:flex-row gap-12 md:gap-20">
                    <div>
                        <div className="text-[#F96515] text-4xl font-black mb-2">4-in-1</div>
                        <div className="text-[#9CA3AF] text-xs font-bold tracking-widest uppercase">Unified Task Backbone</div>
                    </div>
                    <div>
                        <div className="text-[#F96515] text-4xl font-black mb-2">0.09s</div>
                        <div className="text-[#9CA3AF] text-xs font-bold tracking-widest uppercase">Query Latency</div>
                    </div>
                    <div>
                        <div className="text-[#F96515] text-4xl font-black mb-2">Optical+SAR</div>
                        <div className="text-[#9CA3AF] text-xs font-bold tracking-widest uppercase">Cross-Modal Fusion</div>
                    </div>
                </div>
            </div>

        </section>
    );
}