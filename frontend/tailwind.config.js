/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pitch: {
          dark: '#1a1a1a',
          darker: '#0a0a0a',
        },
        accent: {
          neon: '#39ff14',
          yellow: '#ffd700',
        },
        verdict: {
          offside: '#ff3b3b',
          onside: '#39ff14',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
