import { useEffect, useMemo, useState } from 'react'
import { BrainCircuit } from 'lucide-react'
import { useAppStore } from '../store/useAppStore.js'
import { getPipelineRuns } from '../api/redrobApi.js'

const auditSeeds = [
  'Initializing neural pathways...',
  'Loading candidate database...',
  'Parsing job description vectors...',
  'Normalizing career timeline artifacts...',
  'Running L1-L7 survival heuristics...',
  'Resolving duplicate identity clusters...',
  'Scoring collaboration entropy...',
  'Synthesizing final shortlist...'
]

const POLL_INTERVAL_MS = 1500

export default function ProcessingPage() {
  const completeProcessing = useAppStore((state) => state.completeProcessing)
  const failProcessing = useAppStore((state) => state.failProcessing)
  const setProcessingProgress = useAppStore((state) => state.setProcessingProgress)
  const [progress, setProgress] = useState(7)
  const [runnerActive, setRunnerActive] = useState(false)
  const [runnerY, setRunnerY] = useState(0)
  const [score, setScore] = useState(0)

  const audit = useMemo(() => {
    const visibleCount = Math.min(auditSeeds.length, Math.max(2, Math.ceil(progress / 14)))
    return auditSeeds.slice(0, visibleCount)
  }, [progress])

  useEffect(() => {
    setProcessingProgress(progress)
  }, [progress, setProcessingProgress])

  // Real pipeline status polling — advances/completes based on rank.py actually finishing.
  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const runs = await getPipelineRuns()
        const run = runs?.[0]
        if (!run || cancelled) return

        if (run.status === 'running') {
          setProgress((current) => Math.min(92, current + 4))
        } else if (run.status === 'failed') {
          failProcessing('The ranking pipeline failed. Check that your file matches the expected candidate schema.')
        } else {
          setProgress(100)
        }
      } catch {
        // transient network hiccup during polling — try again next tick
      }
    }

    poll()
    const timer = window.setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [failProcessing])

  useEffect(() => {
    if (progress >= 100) {
      const finish = window.setTimeout(() => completeProcessing(), 600)
      return () => window.clearTimeout(finish)
    }
    return undefined
  }, [completeProcessing, progress])

  useEffect(() => {
    const onKey = (event) => {
      if (event.code === 'Space') {
        event.preventDefault()
        jump()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  function jump() {
    setRunnerActive(true)
    setRunnerY(1)
    window.setTimeout(() => setRunnerY(0), 340)
    setScore((current) => current + 14)
  }

  return (
    <main className="processing-page">
      <section className="processing-shell">
        <header className="processing-header">
          <div className="processing-brand">
            <div className="processing-icon">
              <BrainCircuit size={28} aria-hidden="true" />
            </div>
            <div>
              <h1>Fuzzy AI</h1>
              <p>Reasoning in progress</p>
            </div>
          </div>

          <div className="processing-load">
            <div>
              <span>System Load: Stable</span>
              <strong>{progress}%</strong>
            </div>
            <div className="processing-bar">
              <span style={{ width: `${progress}%` }} />
            </div>
          </div>
        </header>

        <aside className="audit-panel">
          <div className="audit-head">
            <span>Live Audit</span>
            <em />
          </div>
          <div className="audit-list">
            {audit.map((item, index) => (
              <p key={item}>
                <span>[12:26:{String(6 + index * 3).padStart(2, '0')}]</span>
                {item}
              </p>
            ))}
          </div>
        </aside>

        <section className="game-zone" onClick={jump}>
          <div className="game-score">
            <span>Score</span>
            <strong>{String(score).padStart(5, '0')}</strong>
          </div>
          <div className="runner-lane">
            <div className={`runner-player ${runnerY ? 'jumping' : ''}`}>^</div>
            <div className={`runner-obstacle ${runnerActive ? 'active' : ''}`} />
          </div>
          <div className="game-copy">
            <h2>Neural Runner</h2>
            <p>{runnerActive ? 'Keep jumping while the shortlist compiles' : 'Press space or click to start'}</p>
          </div>
          <span className="game-hint">Spacebar to jump</span>
        </section>

        <footer className="processing-footer">
          <span>Welcome</span>
          <span>Command Center</span>
          <strong>Session ID: 88-XRAY</strong>
        </footer>
      </section>
    </main>
  )
}
