import { useParams } from 'react-router-dom'
import SVGViewer from '../components/SVGViewer'

export default function IncidentDetail({ incident: propIncident }) {
  const { id } = useParams()

  if (!propIncident) {
    return (
      <div className="max-w-5xl mx-auto">
        <a href="/" className="text-gray-400 hover:text-white mb-6 inline-block">
          ← Back to Console
        </a>
        <div className="card text-center py-12">
          <h2 className="text-xl font-bold mb-2">Incident #{id}</h2>
          <p className="text-gray-400">No incident data available.</p>
          <p className="text-gray-500 text-sm mt-4">
            Analyze a frame from the Match Console to create incidents.
          </p>
        </div>
      </div>
    )
  }

  const incident = propIncident
  const isOffside = incident.decision === 'OFFSIDE'
  
  // Calibration quality color
  const calQualityColors = {
    good: 'text-green-400',
    poor: 'text-yellow-400',
    fallback: 'text-orange-400',
    failed: 'text-gray-400'
  }

  return (
    <div className="max-w-5xl mx-auto">
      <a href="/" className="text-gray-400 hover:text-white mb-6 inline-block">
        ← Back to Console
      </a>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <span className={`text-sm font-bold uppercase ${isOffside ? 'text-red-400' : 'text-green-400'}`}>
                {isOffside ? '🚩 Offside' : '⚽ Goal Check'}
              </span>
              {incident.timestamp && (
                <span className="text-gray-400 text-sm">{incident.timestamp}</span>
              )}
            </div>

            {incident.annotated_image_url && (
              <img
                src={incident.annotated_image_url}
                alt="Annotated frame"
                className="w-full rounded-lg border border-gray-700 mb-4"
              />
            )}

            <div className="flex gap-2">
              <button className="btn-secondary flex-1">📥 Download Report</button>
              <button className="btn-secondary flex-1">🖨 Print</button>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold">AI Verdict</h3>
              <span className={`verdict-badge ${isOffside ? 'verdict-offside' : 'verdict-onside'}`}>
                {incident.decision}
              </span>
            </div>

            <div className="flex items-center gap-4 mb-4">
              <span className="text-gray-400">Confidence:</span>
              <span className="text-2xl font-bold text-accent-neon">
                {(incident.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-red-900/30 border border-red-800 rounded-lg p-4">
                <p className="text-red-400 text-sm mb-1">Attacker</p>
                <p className="font-mono text-lg">
                  x: {incident.attacker_foot?.x?.toFixed(1)}, y: {incident.attacker_foot?.y?.toFixed(1)}
                </p>
              </div>
              <div className="bg-blue-900/30 border border-blue-800 rounded-lg p-4">
                <p className="text-blue-400 text-sm mb-1">2nd Last Defender</p>
                <p className="font-mono text-lg">
                  x: {incident.defender_foot?.x?.toFixed(1)}, y: {incident.defender_foot?.y?.toFixed(1)}
                </p>
              </div>
            </div>

            {/* New: Calibration Quality and Offside Margin */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">Calibration Quality</p>
                <p className={`font-semibold uppercase ${calQualityColors[incident.calibration_quality] || 'text-gray-400'}`}>
                  {incident.calibration_quality || 'unknown'}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-gray-400 text-sm mb-1">Offside Margin</p>
                <p className={`font-mono font-semibold ${isOffside ? 'text-red-400' : 'text-green-400'}`}>
                  {incident.offside_margin_meters?.toFixed(2) || '0.00'} m
                </p>
              </div>
            </div>
          </div>

          {incident.svg_url && (
            <div className="card">
              <h3 className="text-lg font-bold mb-4">Top-Down Visualization</h3>
              <div className="bg-green-800 rounded-lg p-2">
                <img
                  src={incident.svg_url}
                  alt="Pitch diagram"
                  className="w-full"
                />
              </div>
            </div>
          )}

          {incident.explanation && (
            <div className="card">
              <h3 className="text-lg font-bold mb-4">AI Explanation</h3>
              <p className="text-gray-300">{incident.explanation}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}