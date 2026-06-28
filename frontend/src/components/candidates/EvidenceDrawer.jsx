import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, Calendar, Share2, X } from 'lucide-react'
import ProgressBar from '../ui/ProgressBar.jsx'
import { scheduleInterview } from '../../api/redrobApi.js'
import { useCandidate } from '../../hooks/useCandidates.js'
import { useAppStore } from '../../store/useAppStore.js'

export default function EvidenceDrawer() {
  const activeCandidateId = useAppStore((state) => state.activeCandidateId)
  const drawerOpen = useAppStore((state) => state.drawerOpen)
  const closeDrawer = useAppStore((state) => state.closeDrawer)
  const setActivePage = useAppStore((state) => state.setActivePage)
  const { data: candidate } = useCandidate(activeCandidateId)

  async function handleSchedule() {
    await scheduleInterview(activeCandidateId)
    setActivePage('detail')
  }

  return (
    <AnimatePresence>
      {drawerOpen && candidate ? (
        <motion.aside
          className="evidence-drawer"
          initial={{ x: 420, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 420, opacity: 0 }}
          transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="drawer-header">
            <div>
              <h2>Evidence Drawer</h2>
              <p>Profile Integrity Check: {candidate.name}</p>
            </div>
            <button className="icon-button" type="button" onClick={closeDrawer} aria-label="Close drawer">
              <X size={18} aria-hidden="true" />
            </button>
          </div>

          <div className="drawer-scroll">
            <section className="logic-breakdown">
              <h3>Logic Breakdown (L2-L7)</h3>
              <div className="logic-card-grid">
                {Object.entries(candidate.logic_scores)
                  .slice(0, 4)
                  .map(([layer, value]) => (
                    <article className="logic-score-card" key={layer}>
                      <p>{layer.toUpperCase()}: {layer === 'l2' ? 'Knowledge' : layer === 'l3' ? 'Reasoning' : layer === 'l4' ? 'Technical' : 'Systems'}</p>
                      <div>
                        <strong>{value}</strong>
                        <ProgressBar value={value} />
                      </div>
                    </article>
                  ))}
              </div>
            </section>

            <section className="confidence-card">
              <div>
                <p>Model Confidence</p>
                <span>High-density data verification complete</span>
              </div>
              <strong>98%</strong>
            </section>

            <section className="synthesis-card">
              <h3>Synthesis Fit</h3>
              <blockquote>
                "{candidate.name.split(' ')[0]} demonstrates an exceptional grasp of transformer architectures,
                specifically optimized for edge-device deployment. Track record signals high-confidence velocity."
              </blockquote>
            </section>

            <section>
              <h3>Latest Verified Signals</h3>
              <ul className="evidence-list">
                {candidate.evidence.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>
          </div>

          <div className="drawer-actions">
            <button className="primary-action wide" type="button" onClick={handleSchedule}>
              <Calendar size={16} aria-hidden="true" />
              <span>Schedule Interview</span>
            </button>
            <button className="ghost-button" type="button" onClick={() => setActivePage('detail')}>
              <ArrowRight size={16} aria-hidden="true" />
              <span>Dossier</span>
            </button>
            <button className="ghost-button icon-only" type="button" aria-label="Share candidate">
              <Share2 size={16} aria-hidden="true" />
            </button>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  )
}
