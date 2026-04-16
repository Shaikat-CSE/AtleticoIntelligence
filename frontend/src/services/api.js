import axios from 'axios'

const API_BASE = '/api/v1'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
})

export const analyzeFrame = async (imageFile, goalDirection = 'right') => {
  const formData = new FormData()
  formData.append('image_file', imageFile)
  formData.append('goal_direction', goalDirection)

  const response = await api.post('/analyze-frame', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

export const generateVisual = async (data) => {
  const response = await api.post('/generate-visual', data)
  return response.data
}

// DEPRECATED - Removed endpoints
export const analyzeFrameVision = async () => {
  throw new Error('LLM Vision endpoint removed. Use analyzeFrame.')
}

export const analyzeFrameHybrid = async () => {
  throw new Error('Hybrid endpoint removed. Use analyzeFrame.')
}

export default api