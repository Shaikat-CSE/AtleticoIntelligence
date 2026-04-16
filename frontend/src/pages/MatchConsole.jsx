import { useState } from 'react'
import VideoPlayer from '../components/VideoPlayer'
import VerdictDisplay from '../components/VerdictDisplay'

export default function MatchConsole() {
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])

  const handleAnalysisComplete = (analysisResult) => {
    setResult(analysisResult)
    if (analysisResult.decision !== 'UNKNOWN') {
      setHistory(prev => [{
        id: Date.now(),
        ...analysisResult,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev])
    }
  }

  const handleCloseVerdict = () => {
    setResult(null)
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <VideoPlayer onAnalysisComplete={handleAnalysisComplete} />
        </div>

        <div>
          <div className="card">
            <h3 className="text-lg font-bold mb-4">Analysis History</h3>
            {history.length === 0 ? (
              <p className="text-gray-500 text-sm">No analyses yet. Upload content to begin.</p>
            ) : (
              <div className="space-y-3">
                {history.map((item) => (
                  <div key={item.id} className="bg-gray-800 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className={`verdict-badge text-sm ${
                        item.decision === 'OFFSIDE' ? 'verdict-offside' : 'verdict-onside'
                      }`}>
                        {item.decision}
                      </span>
                      <span className="text-gray-500 text-xs">{item.timestamp}</span>
                    </div>
                    <p className="text-gray-400 text-xs">
                      Confidence: {(item.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card mt-4">
            <h3 className="text-lg font-bold mb-2">How to Use</h3>
            <ol className="text-gray-400 text-sm space-y-2 list-decimal list-inside">
              <li>Upload a video or image</li>
              <li>For video: scrub to frame and click "Review This Frame"</li>
              <li>For image: click "Analyze This Image"</li>
              <li>View AI verdict and visualization</li>
            </ol>
          </div>
        </div>
      </div>

      {result && <VerdictDisplay result={result} onClose={handleCloseVerdict} />}
    </div>
  )
}
