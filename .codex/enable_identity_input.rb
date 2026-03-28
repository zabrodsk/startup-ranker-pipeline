path = "/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/web/static/index.html"
text = File.read(path)

text.sub!(
  "  sendBtn.disabled = supabaseAuthState === 'sending' || configMissing;\n  emailInput.disabled = configMissing;\n",
  "  sendBtn.disabled = supabaseAuthState === 'sending';\n  emailInput.disabled = false;\n",
)

text.sub!(
  "      throw new Error('Supabase auth is not configured.');\n",
  "      throw new Error('Magic-link sign-in is not configured yet in this local environment.');\n",
)

File.write(path, text)
