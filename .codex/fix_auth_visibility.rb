path = "/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/web/static/index.html"
text = File.read(path)

text.sub!(
  "  const config = getSupabaseAuthConfig();\n  if (!config?.configured) {\n    section.style.display = 'none';\n    return;\n  }\n\n  section.style.display = 'block';\n",
  "  const config = getSupabaseAuthConfig();\n  const hasUrl = !!config?.url;\n  const isConfigured = !!config?.configured;\n  if (!hasUrl) {\n    section.style.display = 'none';\n    return;\n  }\n\n  section.style.display = 'block';\n",
)

text.sub!(
  "  const authReady = !!getSupabaseAccessToken();\n",
  "  const authReady = !!getSupabaseAccessToken();\n  const configMissing = hasUrl && !isConfigured;\n",
)

text.sub!(
  "  sendBtn.disabled = supabaseAuthState === 'sending';\n",
  "  sendBtn.disabled = supabaseAuthState === 'sending' || configMissing;\n  emailInput.disabled = configMissing;\n",
)

text.sub!(
  "  } else if (supabaseAuthState === 'error' && supabaseAuthError) {\n",
  "  } else if (configMissing) {\n    status.textContent = 'Local Supabase browser auth is not configured yet. Add SUPABASE_ANON_KEY to enable magic-link sign-in.';\n    signedOut.style.display = 'block';\n    signedIn.style.display = 'none';\n  } else if (supabaseAuthState === 'error' && supabaseAuthError) {\n",
)

File.write(path, text)

env_path = "/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/.env"
env_text = File.read(env_path)
unless env_text.include?("SUPABASE_AUTH_REDIRECT_URL=")
  env_text = "#{env_text}\nSUPABASE_AUTH_REDIRECT_URL=http://localhost:8005/\n"
  File.write(env_path, env_text)
end
