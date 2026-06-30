import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

function ScoreOrb({ score }) {
  const meshRef = useRef()
  const intensity = Math.max(0.45, score / 100)
  const color = score >= 85 ? '#8ad5c0' : score >= 70 ? '#ffb86d' : '#ff9999'
  const threeColor = useMemo(() => new THREE.Color(color), [color])

  useFrame(({ clock }) => {
    if (!meshRef.current) return
    meshRef.current.rotation.y += 0.012
    meshRef.current.material.emissiveIntensity = intensity * (0.7 + 0.3 * Math.sin(clock.elapsedTime * 2))
  })

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.78, 32, 32]} />
      <meshStandardMaterial
        color={threeColor}
        emissive={threeColor}
        emissiveIntensity={intensity}
        metalness={0.5}
        roughness={0.16}
      />
    </mesh>
  )
}

export default function ScoreOrb3D({ score }) {
  return (
    <div className="score-orb" aria-label={`Score ${score}`}>
      <Canvas camera={{ position: [0, 0, 3.2], fov: 38 }} dpr={[1, 1.5]} gl={{ alpha: true, antialias: true }}>
        <ambientLight intensity={0.45} />
        <pointLight position={[2, 2, 2]} color="#ffffff" intensity={1.1} />
        <pointLight position={[-2, -1, 2]} color="#5846c8" intensity={0.45} />
        <ScoreOrb score={score} />
      </Canvas>
      <span className="score-orb-value">{score}</span>
    </div>
  )
}
