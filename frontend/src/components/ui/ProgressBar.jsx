export default function ProgressBar({ value, tone = 'primary' }) {
  return (
    <div className="progress-track" aria-label={`${value}%`}>
      <div className={`progress-fill progress-${tone}`} style={{ width: `${value}%` }} />
    </div>
  )
}
