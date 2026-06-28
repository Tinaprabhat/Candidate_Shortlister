import { motion } from 'framer-motion'

export default function LogicMatrix({ scores }) {
  const bars = Object.values(scores || {})

  return (
    <div className="logic-matrix">
      {bars.map((height, index) => (
        <motion.div
          key={`${height}-${index}`}
          className="logic-bar-shell"
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ delay: index * 0.04, type: 'spring', stiffness: 260, damping: 20 }}
          style={{ height: `${height}%`, transformOrigin: 'bottom' }}
        >
          <div className="logic-bar-fill" />
        </motion.div>
      ))}
    </div>
  )
}
