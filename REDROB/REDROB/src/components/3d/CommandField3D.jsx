import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'

function SignalField() {
  const pointsRef = useRef()
  const ringRef = useRef()
  const positions = useMemo(() => {
    const values = []
    for (let i = 0; i < 160; i += 1) {
      const x = (Math.random() - 0.5) * 9
      const y = (Math.random() - 0.5) * 4
      const z = (Math.random() - 0.5) * 3
      values.push(x, y, z)
    }
    return new Float32Array(values)
  }, [])

  useFrame(({ clock }) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y = clock.elapsedTime * 0.06
      pointsRef.current.rotation.x = Math.sin(clock.elapsedTime * 0.4) * 0.06
    }
    if (ringRef.current) {
      ringRef.current.rotation.z = clock.elapsedTime * 0.18
      ringRef.current.rotation.x = 1.1
    }
  })

  return (
    <>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial color="#8ad5c0" size={0.035} transparent opacity={0.72} />
      </points>
      <mesh ref={ringRef} position={[2.6, -0.2, -0.4]}>
        <torusGeometry args={[0.78, 0.02, 12, 64]} />
        <meshStandardMaterial color="#5846c8" emissive="#5846c8" emissiveIntensity={0.35} />
      </mesh>
      <mesh position={[2.6, -0.2, -0.4]}>
        <icosahedronGeometry args={[0.28, 1]} />
        <meshStandardMaterial color="#8ad5c0" emissive="#005243" emissiveIntensity={0.45} />
      </mesh>
    </>
  )
}

export default function CommandField3D() {
  return (
    <div className="command-field" aria-hidden="true">
      <Canvas camera={{ position: [0, 0, 5], fov: 50 }} dpr={[1, 1.5]} gl={{ alpha: true, antialias: true }}>
        <ambientLight intensity={0.65} />
        <pointLight position={[3, 2, 3]} color="#8ad5c0" intensity={1.4} />
        <pointLight position={[-3, -1, 2]} color="#5846c8" intensity={0.8} />
        <SignalField />
      </Canvas>
    </div>
  )
}
