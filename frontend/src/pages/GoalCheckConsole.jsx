import { useState } from 'react'
import GoalCheckPlayer from '../components/GoalCheckPlayer'
import GoalCheckDisplay from '../components/GoalCheckDisplay'

export default function GoalCheckConsole() {
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])

  const handleAnalysisComplete = (analysisResult) => {
    setResult(analysisResult)
    if (analysisResult.decision !== 'UNKNOWN') {
      setHistory((prev) => [{
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
          <GoalCheckPlayer onAnalysisComplete={handleAnalysisComplete} />
        </div>

        <div>
          <div className="card">
            <h3 className="text-lg font-bold mb-4">Goal Check History</h3>
            {history.length === 0 ? (
              <p className="text-gray-500 text-sm">No goal checks yet. Upload content to begin.</p>
            ) : (
              <div className="space-y-3">
                {history.map((item) => (
                  <div key={item.id} className="bg-gray-800 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className={`verdict-badge text-sm ${
                        item.decision === 'GOAL' ? 'verdict-onside' : 'verdict-offside'
                      }`}>
                        {item.decision}
                      </span>
                      <span className="text-gray-500 text-xs">{item.timestamp}</span>
                    </div>
                    <p className="text-gray-400 text-xs">
                      Confidence: {(item.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="text-gray-500 text-xs mt-1">
                      Checked goal: {item.goal_direction}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card mt-4">
            <h3 className="text-lg font-bold mb-2">How Goal Check Works</h3>
            <ol className="text-gray-400 text-sm space-y-2 list-decimal list-inside">
              <li>Upload a frame or freeze a video at the goal event</li>
              <li>Select the left or right goal to review</li>
              <li>Run goal check to detect the ball and estimate the goal line</li>
              <li>Review the annotated frame and margin result</li>
            </ol>
          </div>
        </div>
      </div>

      {result && <GoalCheckDisplay result={result} onClose={handleCloseVerdict} />}
    </div>
  )
}
