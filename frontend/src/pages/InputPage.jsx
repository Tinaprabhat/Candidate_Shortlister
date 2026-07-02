import { useRef, useState } from 'react'
import { BadgeCheck, FileJson, LogIn } from 'lucide-react'
import Sparkles from '../components/ui/Sparkles.jsx'
import { useAppStore } from '../store/useAppStore.js'
import { uploadCandidates } from '../api/redrobApi.js'

function UploadBox({ icon: Icon, title, subtitle, accept, onFile, file }) {
  const inputRef = useRef(null)

  function handleDrop(event) {
    event.preventDefault()
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) onFile(dropped)
  }

  return (
    <button
      className={`input-upload-box ${file ? 'has-file' : ''}`}
      type="button"
      onClick={() => inputRef.current?.click()}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
      <Icon size={32} aria-hidden="true" />
      <strong>{file ? file.name : title}</strong>
      <span>{file ? `${Math.max(1, Math.round(file.size / 1024))} KB queued` : subtitle}</span>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={(event) => {
          const selected = event.target.files?.[0]
          if (selected) onFile(selected)
        }}
      />
    </button>
  )
}

export default function InputPage() {
  const startProcessing = useAppStore((state) => state.startProcessing)
  const failProcessing = useAppStore((state) => state.failProcessing)
  const sandboxError = useAppStore((state) => state.sandboxError)
  const [candidatesFile, setCandidatesFile] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  const ready = Boolean(candidatesFile) && !submitting
  const statusText = submitting
    ? 'Uploading to the ranking pipeline...'
    : candidatesFile
      ? 'Candidate file ready. Job description is preloaded for this challenge.'
      : 'Awaiting a candidates.jsonl file (one profile per line, up to 100 rows).'

  async function handleStart() {
    if (!candidatesFile) return
    setSubmitting(true)
    try {
      const result = await uploadCandidates(candidatesFile)
      startProcessing({ candidates: candidatesFile, jobDescription: null }, result.run_id)
    } catch (error) {
      const detail = error.response?.data?.detail || 'Upload failed. Check the file and try again.'
      failProcessing(detail)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <main className="input-page">
      <Sparkles
        id="redrob-input-stars"
        className="input-stars"
        background="transparent"
        minSize={0.8}
        maxSize={2.4}
        particleDensity={260}
        particleColor="#f4fffb"
        speed={0.55}
      />
      <div className="input-frame-back" aria-hidden="true" />
      <section className="input-console">
        <header className="input-welcome">
          <div className="welcome-mark">W</div>
          <div>
            <h1>Welcome</h1>
            <p>Fuzzy AI Reasoning Shortlister</p>
          </div>
        </header>

        <div className="input-pill">
          <BadgeCheck size={20} aria-hidden="true" />
          <span>Get your best suited candidates</span>
        </div>

        <div className="input-upload-group">
          <label>Upload candidates (.jsonl)</label>
          <UploadBox
            icon={FileJson}
            title="Drag & drop or browse .jsonl file"
            subtitle="One candidate profile JSON per line, up to 100 rows"
            accept=".jsonl,.json"
            file={candidatesFile}
            onFile={setCandidatesFile}
          />
        </div>

        <button className="input-start" type="button" disabled={!ready} onClick={handleStart}>
          <LogIn size={24} aria-hidden="true" />
          <span>{submitting ? 'Uploading...' : 'Start Shortlisting'}</span>
        </button>

        <footer className="input-footer">
          <span>{sandboxError || statusText}</span>
          <span>v 2.0.44-stable</span>
        </footer>
      </section>
    </main>
  )
}
