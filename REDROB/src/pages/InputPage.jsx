import { useMemo, useRef, useState } from 'react'
import { BadgeCheck, FileArchive, FileText, LogIn } from 'lucide-react'
import Sparkles from '../components/ui/Sparkles.jsx'
import { useAppStore } from '../store/useAppStore.js'

function UploadBox({ type, icon: Icon, title, subtitle, accept, onFile, file }) {
  const inputRef = useRef(null)

  function handleDrop(event) {
    event.preventDefault()
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) onFile(type, dropped)
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
          if (selected) onFile(type, selected)
        }}
      />
    </button>
  )
}

export default function InputPage() {
  const startProcessing = useAppStore((state) => state.startProcessing)
  const [files, setFiles] = useState({
    candidates: null,
    jobDescription: null
  })

  const ready = Boolean(files.candidates && files.jobDescription)
  const statusText = useMemo(() => {
    if (ready) return 'Candidate archive and job description locked.'
    if (files.candidates) return 'Candidate archive ready. Add job description.'
    if (files.jobDescription) return 'Job description ready. Add candidate archive.'
    return 'Awaiting candidate archive and job description.'
  }, [files, ready])

  function handleFile(type, file) {
    setFiles((current) => ({ ...current, [type]: file }))
  }

  function handleStart() {
    startProcessing(files)
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
          <label>Upload your candidates folder</label>
          <UploadBox
            type="candidates"
            icon={FileArchive}
            title="Drag & drop or browse zip file"
            subtitle="Fuzzy AI reasoning shortlister"
            accept=".zip"
            file={files.candidates}
            onFile={handleFile}
          />
        </div>

        <div className="input-upload-group">
          <label>Upload job description</label>
          <UploadBox
            type="jobDescription"
            icon={FileText}
            title="Drag & drop or browse PDF/DOCX"
            subtitle="Max size: 10MB"
            accept=".pdf,.doc,.docx"
            file={files.jobDescription}
            onFile={handleFile}
          />
        </div>

        <button className="input-start" type="button" disabled={!ready} onClick={handleStart}>
          <LogIn size={24} aria-hidden="true" />
          <span>Start Shortlisting</span>
        </button>

        <footer className="input-footer">
          <span>{statusText}</span>
          <span>v 2.0.44-stable</span>
        </footer>
      </section>
    </main>
  )
}
