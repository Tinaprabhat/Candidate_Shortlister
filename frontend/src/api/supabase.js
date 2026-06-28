import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

export const hasSupabaseConfig = Boolean(
  supabaseUrl &&
    supabaseAnonKey &&
    !supabaseUrl.includes('xxxx') &&
    !supabaseAnonKey.includes('replace-with')
)

export const supabase = hasSupabaseConfig
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null

export async function signIn(email, password) {
  if (supabase) {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password
    })

    if (error) throw error
    return {
      token: data.session.access_token,
      user: data.user
    }
  }

  if (!email || !password) {
    throw new Error('Enter an email and password.')
  }

  return {
    token: `local-preview-${Date.now()}`,
    user: {
      id: 'local-admin',
      email
    }
  }
}

export async function signOut() {
  if (supabase) {
    await supabase.auth.signOut()
  }
}
