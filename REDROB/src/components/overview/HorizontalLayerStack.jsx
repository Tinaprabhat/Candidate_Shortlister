import { useState } from 'react'
import { motion } from 'framer-motion'

const initialLayers = [
  { id: 'L1', title: 'Input Layer', value: '100k', caption: 'Candidates' },
  { id: 'L2', title: 'Identity Gate', value: '87k', caption: 'Validated' },
  { id: 'L3', title: 'Resume Parse', value: '61k', caption: 'Structured' },
  { id: 'L4', title: 'Skill Match', value: '34k', caption: 'Mapped' },
  { id: 'L5', title: 'Experience Fit', value: '18k', caption: 'Weighted' },
  { id: 'L6', title: 'Integrity Scan', value: '8.7k', caption: 'Verified' },
  { id: 'L7', title: 'Reasoning Core', value: '1.9k', caption: 'Ranked' },
  { id: 'L8', title: 'Human Review', value: '842', caption: 'Queued' },
  { id: 'L9', title: 'Output', value: '214', caption: 'Shortlisted' }
]

export default function HorizontalLayerStack() {
  const [layers, setLayers] = useState(initialLayers)

  function moveToEnd(index) {
    setLayers((current) => [...current.slice(index + 1), current[index], ...current.slice(0, index)])
  }

  return (
    <section className="panel horizontal-layer-panel">
      <div className="section-heading">
        <h2>Pipeline Layer Stack</h2>
        <p>Horizontal evaluation depth</p>
      </div>

      <div className="horizontal-card-stage" data-animation="horizontal-card-stack">
        {layers.map((layer, index) => {
          const isFront = index === 0
          const offset = index * 24
          const scale = 1 - index * 0.025
          const brightness = Math.max(0.32, 1 - index * 0.08)

          return (
            <motion.button
              type="button"
              key={layer.id}
              className={`layer-card ${isFront ? 'front' : ''}`}
              style={{
                zIndex: layers.length - index,
                cursor: isFront ? 'grab' : 'pointer'
              }}
              animate={{
                x: offset,
                y: index * 4,
                scale,
                filter: `brightness(${brightness})`
              }}
              transition={{ type: 'spring', stiffness: 170, damping: 26 }}
              drag={isFront ? 'x' : false}
              dragConstraints={{ left: -80, right: 80 }}
              dragMomentum={false}
              onDragEnd={() => moveToEnd(index)}
              onClick={() => moveToEnd(index)}
              whileDrag={
                isFront
                  ? {
                      scale: scale + 0.04,
                      rotate: -1.5,
                      cursor: 'grabbing'
                    }
                  : {}
              }
            >
              <span>{layer.title}</span>
              <strong>{layer.value}</strong>
              <em>{layer.caption}</em>
              <div className="layer-card-line" />
              <small>SEQ_ACTIVE_{layer.id}</small>
            </motion.button>
          )
        })}
      </div>

      <div className="layer-stack-footer">
        <span>Layers L1-L9</span>
        <strong>Sequential Filtering Active</strong>
      </div>
    </section>
  )
}
