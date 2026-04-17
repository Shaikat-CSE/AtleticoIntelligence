export default function VerdictDisplay({ result, onClose }) {
  if (!result) return null

  const isOffside = result.decision === 'OFFSIDE'
  const attackingTeamInfo = result[`${result.attacking_team}_info`]
  const defendingTeamInfo = result[`${result.defending_team}_info`]
  const isApproximateColor = (teamInfo) => (
    Boolean(teamInfo?.color_warning) || (teamInfo?.color_confidence ?? 1) < 0.58
  )
  const getTeamLabel = (teamInfo, fallback) => {
    if (!teamInfo) return fallback
    return `${isApproximateColor(teamInfo) ? 'Approx. ' : ''}${teamInfo.color_name}`
  }
  const getConfidenceLabel = (confidence) => {
    if (confidence == null) return null
    if (confidence >= 0.78) return 'High confidence'
    if (confidence >= 0.58) return 'Medium confidence'
    return 'Low confidence'
  }
  
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-6">
      <div className="card max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="flex justify-between items-start mb-6">
          <div className="flex items-center gap-4">
            <div className={`verdict-badge ${isOffside ? 'verdict-offside' : 'verdict-onside'}`}>
              {isOffside ? '🚩 OFFSIDE' : '✅ ONSIDE'}
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
            <h4 className="text-sm font-semibold text-gray-400 mb-2">Annotated Frame</h4>
            <img
              src={result.annotated_image_url}
              alt="Annotated frame"
              className="w-full rounded-lg border border-gray-700"
            />
          </div>
        )}

        {result.svg_url && (
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-400 mb-2">Top-Down View</h4>
            <div className="bg-green-800 rounded-lg p-4">
              <img
                src={`${result.svg_url}?t=${Date.now()}`}
                alt="Pitch diagram"
                className="w-full max-h-64 mx-auto"
              />
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-red-900/30 border border-red-800 rounded-lg p-4">
            <p className="text-red-400 text-sm mb-1">Attacker Position</p>
            <p className="font-mono">
              x: {result.attacker_foot?.x?.toFixed(1)}, y: {result.attacker_foot?.y?.toFixed(1)}
            </p>
          </div>
          <div className="bg-blue-900/30 border border-blue-800 rounded-lg p-4">
            <p className="text-blue-400 text-sm mb-1">Defender Position</p>
            <p className="font-mono">
              x: {result.defender_foot?.x?.toFixed(1)}, y: {result.defender_foot?.y?.toFixed(1)}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Attacking Side</p>
            <p className="font-semibold">{getTeamLabel(attackingTeamInfo, result.attacking_team)}</p>
            {attackingTeamInfo?.color_confidence != null && (
              <p className="text-xs text-gray-400 mt-1">
                {getConfidenceLabel(attackingTeamInfo.color_confidence)} ({(attackingTeamInfo.color_confidence * 100).toFixed(0)}%)
              </p>
            )}
            {attackingTeamInfo?.color_warning && (
              <p className="text-xs text-amber-300 mt-1">{attackingTeamInfo.color_warning}</p>
            )}
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Defending Side</p>
            <p className="font-semibold">{getTeamLabel(defendingTeamInfo, result.defending_team)}</p>
            {defendingTeamInfo?.color_confidence != null && (
              <p className="text-xs text-gray-400 mt-1">
                {getConfidenceLabel(defendingTeamInfo.color_confidence)} ({(defendingTeamInfo.color_confidence * 100).toFixed(0)}%)
              </p>
            )}
            {defendingTeamInfo?.color_warning && (
              <p className="text-xs text-amber-300 mt-1">{defendingTeamInfo.color_warning}</p>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 mb-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <p className="text-gray-400 text-sm mb-1">Offside Margin</p>
            <p className={`font-mono font-semibold ${isOffside ? 'text-red-400' : 'text-green-400'}`}>
              {result.offside_margin_pixels?.toFixed(1) || '0.0'} px
            </p>
          </div>
        </div>

        {result.explanation && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-400 mb-2">AI Explanation</h4>
            <p className="text-gray-300">{result.explanation}</p>
          </div>
        )}

        <div className="flex gap-4 mt-6">
          <button className="btn-secondary flex-1">📥 Download Report</button>
          <button className="btn-primary flex-1" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  )
}
