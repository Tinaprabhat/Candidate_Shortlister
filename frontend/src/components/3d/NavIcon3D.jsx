import { Canvas } from '@react-three/fiber'
import { animated, useSpring } from '@react-spring/three'

function NavMesh({ active }) {
  const { scale, rotY, emissiveIntensity } = useSpring({
    scale: active ? 1.2 : 0.95,
    rotY: active ? 0.32 : 0,
    emissiveIntensity: active ? 0.85 : 0.08,
    config: { tension: 280, friction: 22 }
  })

  return (
    <animated.mesh scale={scale} rotation-y={rotY}>
      <boxGeometry args={[1, 1, 0.2]} />
      <animated.meshStandardMaterial
        color={active ? '#8ad5c0' : '#6f7975'}
        emissive="#005243"
        emissiveIntensity={emissiveIntensity}
        metalness={0.4}
        roughness={0.28}
      />
    </animated.mesh>
  )
}

export default function NavIcon3D({ active }) {
  return (
    <span className="nav-canvas" aria-hidden="true">
      <Canvas camera={{ position: [0, 0, 3], fov: 42 }} dpr={[1, 1.5]} gl={{ alpha: true, antialias: true }}>
        <ambientLight intensity={0.55} />
        <pointLight position={[2, 2, 2]} color="#8ad5c0" intensity={active ? 1.25 : 0.6} />
        <NavMesh active={active} />
      </Canvas>
    </span>
  )
}
