import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import MatchConsole from './pages/MatchConsole'
import IncidentDetail from './pages/IncidentDetail'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-pitch-darker">
        <header className="bg-pitch-dark border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-accent-neon rounded-lg flex items-center justify-center">
                <span className="text-black font-bold text-xl">AI</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">AtleticoIntelligence</h1>
                <p className="text-gray-400 text-sm">AI-Powered Offside Review System</p>
              </div>
            </div>
            <nav className="flex gap-6">
              <a href="/" className="text-gray-300 hover:text-accent-neon transition-colors">Match Console</a>
            </nav>
          </div>
        </header>

        <main className="p-6 max-w-7xl mx-auto">
          <Routes>
            <Route path="/" element={<MatchConsole />} />
            <Route path="/incident/:id" element={<IncidentDetail />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
