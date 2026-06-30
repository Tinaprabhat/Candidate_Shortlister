import { useEffect, useRef, useState } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'

export default function ScrollShowcase({ eyebrow, title, children, compact = false }) {
  const containerRef = useRef(null)
  const [isMobile, setIsMobile] = useState(false)
  const { scrollYProgress } = useScroll({ target: containerRef, offset: ['start end', 'end start'] })

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth <= 768)
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const rotate = useTransform(scrollYProgress, [0, 1], [compact ? 8 : 14, 0])
  const scale = useTransform(scrollYProgress, [0, 1], isMobile ? [0.96, 1] : [1.025, 1])
  const translate = useTransform(scrollYProgress, [0, 1], [20, -28])

  return (
    <section className={`scroll-showcase ${compact ? 'compact' : ''}`} ref={containerRef} data-animation="scroll-showcase">
      <motion.div className="scroll-showcase-heading" style={{ translateY: translate }}>
        <span>{eyebrow}</span>
        <h2>{title}</h2>
      </motion.div>
      <div className="scroll-perspective">
        <motion.div className="scroll-device" style={{ rotateX: rotate, scale }}>
          {children}
        </motion.div>
      </div>
    </section>
  )
}
