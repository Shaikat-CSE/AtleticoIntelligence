import { useRef, useState } from 'react'
import { checkGoal } from '../services/api'

export default function GoalCheckPlayer({ onAnalysisComplete }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [uploadedImageFile, setUploadedImageFile] = useState(null)
  const [goalDirection, setGoalDirection] = useState('right')

  const handleVideoUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return

    setVideoUrl(URL.createObjectURL(file))
    setUploadedImage(null)
    setUploadedImageFile(null)
  }

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (!file) return

    setUploadedImage(URL.createObjectURL(file))
    setUploadedImageFile(file)
    setVideoUrl(null)
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
        if (!blob) {
          resolve(null)
          return
        }
        resolve(new File([blob], 'goal-check-frame.jpg', { type: 'image/jpeg' }))
      }, 'image/jpeg')
    })
  }

  const runGoalCheck = async (file) => {
    if (!file) return

    setIsAnalyzing(true)
    try {
      const result = await checkGoal(file, goalDirection)
      if (onAnalysisComplete) onAnalysisComplete(result)
    } catch (error) {
      console.error('Goal check failed:', error)
      alert(error?.response?.data?.detail || 'Failed to run goal check. Is the backend running?')
    }
    setIsAnalyzing(false)
  }

  const handleCheckImage = async () => {
    await runGoalCheck(uploadedImageFile)
  }

  const handleCheckVideoFrame = async () => {
    const frameFile = await captureFrame()
    await runGoalCheck(frameFile)
  }

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-lg font-bold mb-4">Goal Check Input</h3>

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
              Choose Video File
            </button>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Upload Image Frame</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="goal-check-image-upload"
            />
            <button
              onClick={() => document.getElementById('goal-check-image-upload')?.click()}
              className="btn-secondary w-full"
            >
              Choose Image File
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <label className="block text-sm text-gray-400 mb-2">Goal To Check</label>
        <div className="flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="goalCheckDirection"
              value="left"
              checked={goalDirection === 'left'}
              onChange={(e) => setGoalDirection(e.target.value)}
            />
            <span className="text-yellow-400">Left goal</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="goalCheckDirection"
              value="right"
              checked={goalDirection === 'right'}
              onChange={(e) => setGoalDirection(e.target.value)}
            />
            <span className="text-green-400">Right goal</span>
          </label>
        </div>
        <p className="text-xs text-gray-500 mt-3">
          This review uses only the ball, the selected goal side, and detected goal-side white geometry such as the post and line.
        </p>
      </div>

      {uploadedImage && (
        <div className="card">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-bold">Uploaded Image</h3>
            <span className="text-sm text-gray-400">Goal-line review</span>
          </div>
          <img
            src={uploadedImage}
            alt="Uploaded frame"
            className="w-full rounded-lg border border-gray-700 mb-4"
          />

          <button
            onClick={handleCheckImage}
            disabled={isAnalyzing}
            className="btn-primary w-full"
          >
            {isAnalyzing ? 'Analyzing...' : 'Check Goal'}
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
              {isPlaying ? 'Pause' : 'Play'}
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

          <button
            onClick={handleCheckVideoFrame}
            disabled={isAnalyzing}
            className="btn-primary w-full"
          >
            {isAnalyzing ? 'Analyzing...' : 'Check Current Frame'}
          </button>
        </div>
      )}

      {!videoUrl && !uploadedImage && (
        <div className="card text-center py-12">
          <div className="text-5xl mb-4">GLT</div>
          <h3 className="text-xl font-bold mb-2">No Goal Review Frame Loaded</h3>
          <p className="text-gray-400">
            Upload a freeze frame or scrub a video to a scoring moment, choose the target goal, then run the goal check.
          </p>
        </div>
      )}
    </div>
  )
}
