import { motion } from 'framer-motion'
import { Circle } from 'lucide-react'

function ElegantShape({
  className = '',
  delay = 0,
  width = 400,
  height = 100,
  rotate = 0,
  tone = 'teal'
}) {
  return (
    <motion.div
      className={`elegant-shape ${className}`}
      initial={{ opacity: 0, y: -150, rotate: rotate - 15 }}
      animate={{ opacity: 1, y: 0, rotate }}
      transition={{
        duration: 2.4,
        delay,
        ease: [0.23, 0.86, 0.39, 0.96],
        opacity: { duration: 1.2 }
      }}
    >
      <motion.div
        className="elegant-shape-inner"
        style={{ width, height }}
        animate={{ y: [0, 15, 0] }}
        transition={{
          duration: 12,
          repeat: Number.POSITIVE_INFINITY,
          ease: 'easeInOut'
        }}
      >
        <div className={`elegant-shape-fill tone-${tone}`} />
      </motion.div>
    </motion.div>
  )
}

export default function ShapeLandingHero({
  badge = 'Live ingestion pipeline',
  title1 = 'Systems',
  title2 = 'Overview',
  description = 'Real-time intelligence dashboard monitoring candidate flow, heuristic survival rates, and risk distribution across the active ingestion pipeline.'
}) {
  const fadeUpVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: (index) => ({
      opacity: 1,
      y: 0,
      transition: {
        duration: 1,
        delay: 0.25 + index * 0.16,
        ease: [0.25, 0.4, 0.25, 1]
      }
    })
  }

  return (
    <div className="shape-hero" data-animation="landing-hero">
      <div className="shape-hero-glow" />
      <div className="shape-hero-shapes">
        <ElegantShape delay={0.2} width={540} height={116} rotate={12} tone="teal" className="shape-a" />
        <ElegantShape delay={0.35} width={430} height={104} rotate={-15} tone="violet" className="shape-b" />
        <ElegantShape delay={0.48} width={260} height={70} rotate={-8} tone="amber" className="shape-c" />
        <ElegantShape delay={0.56} width={180} height={54} rotate={20} tone="cyan" className="shape-d" />
      </div>

      <div className="shape-hero-content">
        <motion.div
          className="hero-badge"
          custom={0}
          variants={fadeUpVariants}
          initial="hidden"
          animate="visible"
        >
          <Circle size={8} fill="currentColor" aria-hidden="true" />
          <span>{badge}</span>
        </motion.div>
        <motion.h1 custom={1} variants={fadeUpVariants} initial="hidden" animate="visible">
          <span>{title1}</span>
          <span>{title2}</span>
        </motion.h1>
        <motion.p custom={2} variants={fadeUpVariants} initial="hidden" animate="visible">
          {description}
        </motion.p>
      </div>
    </div>
  )
}
