export default function GoalCheckDisplay({ result, onClose }) {
  if (!result) return null

  const isGoal = result.decision === 'GOAL'
  const isUnknown = result.decision === 'UNKNOWN'
  const badgeClass = isUnknown
    ? 'bg-yellow-500 text-black'
    : isGoal
      ? 'verdict-onside'
      : 'verdict-offside'
  const marginLabel = isUnknown
    ? 'Inconclusive'
    : isGoal
      ? `Ball over line by ${Math.abs(result.goal_margin_pixels || 0).toFixed(1)} px`
      : `Ball short by ${Math.abs(result.goal_margin_pixels || 0).toFixed(1)} px`

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-6">
      <div className="card max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="flex justify-between items-start mb-6">
          <div className="flex items-center gap-4">
            <div className={`verdict-badge ${badgeClass}`}>
              {result.decision}
            </div>
            <div>
              <p className="text-2xl font-bold">{result.decision}</p>
              <p className="text-gray-400">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            </div>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">&times;</button>
        </div>

        {result.annotated_image_url && (
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-400 mb-2">Annotated Goal Review</h4>
            <img
              src={`${result.annotated_image_url}?t=${Date.now()}`}
              alt="Goal check frame"
              className="w-full rounded-lg border border-gray-700"
            />
          </div>
        )}

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Checked Goal</p>
            <p className="font-semibold uppercase">{result.goal_direction}</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Ball Detected</p>
            <p className="font-semibold">{result.ball_detected ? 'Yes' : 'No'}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Goal Geometry Source</p>
            <p className="font-semibold">{result.goal_line_source}</p>
            <p className="text-xs text-gray-500 mt-1">
              Confidence: {(result.goal_line_confidence * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Goal Margin</p>
            <p className={`font-semibold ${isUnknown ? 'text-yellow-300' : isGoal ? 'text-green-400' : 'text-red-400'}`}>
              {marginLabel}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Ball Position</p>
            <p className="font-mono">
              {result.ball_position
                ? `x: ${result.ball_position.x.toFixed(1)}, y: ${result.ball_position.y.toFixed(1)}`
                : 'Unavailable'}
            </p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Goal Line X</p>
            <p className="font-mono">
              {result.goal_line_x != null ? result.goal_line_x.toFixed(1) : 'Unavailable'}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Goalpost Detection</p>
            <p className="font-semibold">
              {result.goalpost_x != null ? 'Detected' : 'Not detected'}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Source: {result.goalpost_source}
            </p>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Goalpost X</p>
            <p className="font-mono">
              {result.goalpost_x != null ? result.goalpost_x.toFixed(1) : 'Unavailable'}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Confidence: {(result.goalpost_confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {result.explanation && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-400 mb-2">Goal-Line Explanation</h4>
            <p className="text-gray-300">{result.explanation}</p>
          </div>
        )}

        <div className="flex gap-4 mt-6">
          <button className="btn-secondary flex-1">Download Report</button>
          <button className="btn-primary flex-1" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  )
}
