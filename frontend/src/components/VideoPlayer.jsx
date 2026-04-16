import { useState, useRef } from 'react'
import { analyzeFrame } from '../services/api'

export default function VideoPlayer({ onAnalysisComplete }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [attackingTeam, setAttackingTeam] = useState('team1')
  const [goalDirection, setGoalDirection] = useState('right')

  const handleVideoUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setVideoUrl(url)
      setUploadedImage(null)
    }
  }

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setUploadedImage(URL.createObjectURL(file))
      setVideoUrl(null)
    }
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration)
    }
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handlePlayPause = () => {
    if (!videoRef.current) return
    if (isPlaying) {
      videoRef.current.pause()
    } else {
      videoRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleScrub = (e) => {
    if (!videoRef.current || !duration) return
    const rect = e.currentTarget.getBoundingClientRect()
    const pos = (e.clientX - rect.left) / rect.width
    videoRef.current.currentTime = pos * duration
  }

  const captureFrame = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' })
        resolve(file)
      }, 'image/jpeg')
    })
  }

  const performAnalysis = async (file) => {
    setIsAnalyzing(true)
    try {
      const result = await analyzeFrame(file, goalDirection, attackingTeam)
      if (onAnalysisComplete) onAnalysisComplete(result)
    } catch (error) {
      console.error('Analysis failed:', error)
      alert('Failed to analyze image. Is the backend running?')
    }
    setIsAnalyzing(false)
  }

  const handleAnalyzeImage = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    await performAnalysis(file)
  }

  const handleAnalyzeFrame = async () => {
    if (!videoRef.current) return
    const frameFile = await captureFrame()
    await performAnalysis(frameFile)
  }

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-lg font-bold mb-4">Upload Content</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Upload Video</label>
            <input
              type="file"
              accept="video/*"
              onChange={handleVideoUpload}
              ref={fileInputRef}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-secondary w-full"
            >
              📹 Choose Video File
            </button>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Upload Image Frame</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="image-upload"
            />
            <button
              onClick={() => document.getElementById('image-upload')?.click()}
              className="btn-secondary w-full"
            >
              🖼 Choose Image File
            </button>
          </div>
        </div>
      </div>

      {uploadedImage && (
        <div className="card">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-bold">Uploaded Image</h3>
            <span className="text-gray-400 text-sm">Click to analyze</span>
          </div>
          <img
            src={uploadedImage}
            alt="Uploaded frame"
            className="w-full rounded-lg border border-gray-700 mb-4"
          />
          
          <div className="mb-4 p-3 bg-gray-800 rounded-lg">
            <label className="block text-sm text-gray-400 mb-2">Select Attacking Team</label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="attackingTeam"
                  value="team1"
                  checked={attackingTeam === 'team1'}
                  onChange={(e) => setAttackingTeam(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-red-400">Team 1 (Red)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="attackingTeam"
                  value="team2"
                  checked={attackingTeam === 'team2'}
                  onChange={(e) => setAttackingTeam(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-blue-400">Team 2 (Blue)</span>
              </label>
            </div>
          </div>

          <div className="mb-4 p-3 bg-gray-800 rounded-lg">
            <label className="block text-sm text-gray-400 mb-2">Attack Direction (Goal Being Attacked)</label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="goalDirection"
                  value="left"
                  checked={goalDirection === 'left'}
                  onChange={(e) => setGoalDirection(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-yellow-400">← LEFT goal (attacking right, toward x=105)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="goalDirection"
                  value="right"
                  checked={goalDirection === 'right'}
                  onChange={(e) => setGoalDirection(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-green-400">RIGHT goal → (attacking left, toward x=0)</span>
              </label>
            </div>
          </div>
          
          <input
            type="file"
            accept="image/*"
            onChange={handleAnalyzeImage}
            className="hidden"
            id="analyze-image"
          />
          <button
            onClick={() => document.getElementById('analyze-image')?.click()}
            disabled={isAnalyzing}
            className="btn-primary w-full disabled:opacity-50"
          >
            {isAnalyzing ? '⏳ Analyzing...' : '🔍 Analyze Frame'}
          </button>
        </div>
      )}

      {videoUrl && (
        <div className="card">
          <div className="relative bg-black rounded-lg overflow-hidden mb-4">
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full aspect-video"
              onClick={handlePlayPause}
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
            />
            <canvas ref={canvasRef} className="hidden" />
          </div>

          <div className="flex items-center gap-4 mb-4">
            <button onClick={handlePlayPause} className="btn-secondary">
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
            <span className="text-gray-400 font-mono">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div
            className="h-2 bg-gray-700 rounded-full cursor-pointer mb-4"
            onClick={handleScrub}
          >
            <div
              className="h-full bg-accent-neon rounded-full transition-all"
              style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
            />
          </div>

          <div className="mb-4 p-3 bg-gray-800 rounded-lg">
            <label className="block text-sm text-gray-400 mb-2">Select Attacking Team</label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="attackingTeamVideo"
                  value="team1"
                  checked={attackingTeam === 'team1'}
                  onChange={(e) => setAttackingTeam(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-red-400">Team 1 (Red)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="attackingTeamVideo"
                  value="team2"
                  checked={attackingTeam === 'team2'}
                  onChange={(e) => setAttackingTeam(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-blue-400">Team 2 (Blue)</span>
              </label>
            </div>
          </div>

          <div className="mb-4 p-3 bg-gray-800 rounded-lg">
            <label className="block text-sm text-gray-400 mb-2">Attack Direction (Goal Being Attacked)</label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="goalDirectionVideo"
                  value="left"
                  checked={goalDirection === 'left'}
                  onChange={(e) => setGoalDirection(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-yellow-400">← LEFT goal (attacking right, toward x=105)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="goalDirectionVideo"
                  value="right"
                  checked={goalDirection === 'right'}
                  onChange={(e) => setGoalDirection(e.target.value)}
                  className="text-accent-neon"
                />
                <span className="text-green-400">RIGHT goal → (attacking left, toward x=0)</span>
              </label>
            </div>
          </div>
          
          <button
            onClick={handleAnalyzeFrame}
            disabled={isAnalyzing}
            className="btn-primary w-full disabled:opacity-50"
          >
            {isAnalyzing ? '⏳ Analyzing...' : '🔍 Review This Frame'}
          </button>
        </div>
      )}

      {!videoUrl && !uploadedImage && (
        <div className="card text-center py-12">
          <div className="text-6xl mb-4">⚽</div>
          <h3 className="text-xl font-bold mb-2">No Content Loaded</h3>
          <p className="text-gray-400">
            Upload a video or image to begin offside analysis
          </p>
        </div>
      )}
    </div>
  )
}