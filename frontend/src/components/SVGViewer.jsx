import { useState, useEffect } from 'react'

export default function SVGViewer({ svgUrl, svgContent: propSvgContent, attackerPos, defenderPos, ballPos, offsideLineX, decision }) {
  const [svgContent, setSvgContent] = useState(propSvgContent)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (propSvgContent) {
      setSvgContent(propSvgContent)
      return
    }

    if (svgUrl) {
      fetch(svgUrl)
        .then(res => res.text())
        .then(setSvgContent)
        .catch(() => setSvgContent(generateFallbackSVG()))
    } else if (attackerPos && defenderPos) {
      setSvgContent(generateFallbackSVG())
    }
  }, [svgUrl, propSvgContent, attackerPos, defenderPos, ballPos, offsideLineX, decision])

  const generateFallbackSVG = () => {
    const scale = 10
    const w = 105 * scale
    const h = 68 * scale

    const ax = attackerPos?.x ? attackerPos.x * scale : 500
    const ay = attackerPos?.y ? attackerPos.y * scale : 340
    const dx = defenderPos?.x ? defenderPos.x * scale : 400
    const dy = defenderPos?.y ? defenderPos.y * scale : 340

    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
      <rect width="100%" height="100%" fill="#2e7d32"/>
      <rect x="10" y="10" width="1030" height="660" fill="#388e3c" stroke="white" stroke-width="2"/>
      <line x1="520" y1="10" x2="520" y2="670" stroke="white" stroke-width="2"/>
      <circle cx="${ax}" cy="${ay}" r="15" fill="red" stroke="white" stroke-width="2"/>
      <circle cx="${dx}" cy="${dy}" r="15" fill="blue" stroke="white" stroke-width="2"/>
      ${offsideLineX ? `<line x1="${offsideLineX * scale}" y1="10" x2="${offsideLineX * scale}" y2="670" stroke="yellow" stroke-width="3"/>` : ''}
    </svg>`
  }

  if (!svgContent) {
    return (
      <div className="bg-green-800 rounded-lg p-4 flex items-center justify-center h-48">
        <span className="text-gray-400">No visualization available</span>
      </div>
    )
  }

  return (
    <div className="bg-green-800 rounded-lg p-2">
      <div dangerouslySetInnerHTML={{ __html: svgContent }} className="w-full" />
    </div>
  )
}
