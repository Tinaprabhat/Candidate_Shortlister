import { useState } from 'react'
import { LockKeyhole, LogIn, ShieldCheck } from 'lucide-react'
import Sparkles from '../components/ui/Sparkles.jsx'
import { hasSupabaseConfig, signIn } from '../api/supabase.js'
import { useAppStore } from '../store/useAppStore.js'

export default function LoginPage() {
  const login = useAppStore((state) => state.login)
  const [email, setEmail] = useState('admin@redrob.ai')
  const [password, setPassword] = useState('preview')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const session = await signIn(email, password)
      login(session)
    } catch (err) {
      setError(err.message || 'Sign-in failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="login-page">
      <Sparkles
        id="redrob-login-sparkles"
        className="login-sparkles"
        background="transparent"
        minSize={0.45}
        maxSize={1.3}
        particleDensity={190}
        particleColor="#8ad5c0"
        speed={0.9}
      />
      <div className="login-beam" aria-hidden="true" />
      <section className="login-panel">
        <div className="login-brand">
          <div className="brand-mark large">RR</div>
          <div>
            <h1>RedRob</h1>
            <p>Command Center</p>
          </div>
        </div>

        <form className="login-form" onSubmit={handleSubmit}>
          <div className="auth-state fit">
            <ShieldCheck size={16} aria-hidden="true" />
            <span>{hasSupabaseConfig ? 'Supabase Auth' : 'Local Preview'}</span>
          </div>

          <label className="field-stack">
            <span>Email</span>
            <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" autoComplete="email" />
          </label>

          <label className="field-stack">
            <span>Password</span>
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              autoComplete="current-password"
            />
          </label>

          {error ? <div className="form-error">{error}</div> : null}

          <button className="primary-action wide" type="submit" disabled={loading}>
            {loading ? <LockKeyhole size={16} aria-hidden="true" /> : <LogIn size={16} aria-hidden="true" />}
            <span>{loading ? 'Authenticating' : 'Enter Command Center'}</span>
          </button>
        </form>
      </section>
    </main>
  )
}
