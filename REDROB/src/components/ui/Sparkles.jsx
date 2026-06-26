import { useCallback, useId, useMemo } from 'react'
import { motion, useAnimation } from 'framer-motion'
import Particles from '@tsparticles/react'

export default function Sparkles({
  id,
  className = '',
  background = 'transparent',
  minSize = 0.4,
  maxSize = 1.2,
  speed = 1,
  particleColor = '#ffffff',
  particleDensity = 160
}) {
  const generatedId = useId()
  const controls = useAnimation()

  const particlesLoaded = useCallback(async () => {
    controls.start({ opacity: 1, transition: { duration: 1 } })
  }, [controls])

  const options = useMemo(
    () => ({
      background: { color: { value: background } },
      fullScreen: { enable: false, zIndex: 1 },
      fpsLimit: 90,
      interactivity: {
        events: {
          onClick: { enable: true, mode: 'push' },
          onHover: { enable: false, mode: 'repulse' },
          resize: { enable: true }
        },
        modes: {
          push: { quantity: 4 },
          repulse: { distance: 200, duration: 0.4 }
        }
      },
      particles: {
        color: { value: particleColor },
        move: {
          direction: 'none',
          enable: true,
          outModes: { default: 'out' },
          random: false,
          speed: { min: 0.1, max: speed },
          straight: false
        },
        number: {
          density: { enable: true, width: 400, height: 400 },
          value: particleDensity
        },
        opacity: {
          value: { min: 0.1, max: 1 },
          animation: {
            enable: true,
            speed: 4,
            sync: false,
            startValue: 'random'
          }
        },
        shape: { type: 'circle' },
        size: {
          value: { min: minSize, max: maxSize }
        }
      },
      detectRetina: true
    }),
    [background, maxSize, minSize, particleColor, particleDensity, speed]
  )

  return (
    <motion.div
      className={`sparkles-core ${className}`}
      data-animation="sparkles"
      animate={controls}
      initial={{ opacity: 0 }}
    >
      <Particles id={id || generatedId} className="sparkles-canvas" particlesLoaded={particlesLoaded} options={options} />
    </motion.div>
  )
}
