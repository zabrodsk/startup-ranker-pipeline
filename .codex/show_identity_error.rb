path = "/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/web/static/index.html"
text = File.read(path)

old = <<~JS
  if (authReady) {
    status.textContent = config.required
      ? 'Signed in. New runs will be stamped with this identity.'
      : 'Signed in. This identity will be attached to any new run you start.';
    userEl.textContent = email && label !== email ? `${label} · ${email}` : label;
  } else if (configMissing) {
    status.textContent = 'Please enter your email address to verify your identity.';
    signedOut.style.display = 'block';
    signedIn.style.display = 'none';
  } else if (supabaseAuthState === 'error' && supabaseAuthError) {
    status.textContent = supabaseAuthError;
  } else if (supabaseAuthState === 'sending') {
JS

new = <<~JS
  if (authReady) {
    status.textContent = config.required
      ? 'Signed in. New runs will be stamped with this identity.'
      : 'Signed in. This identity will be attached to any new run you start.';
    userEl.textContent = email && label !== email ? `${label} · ${email}` : label;
  } else if (supabaseAuthState === 'error' && supabaseAuthError) {
    status.textContent = supabaseAuthError;
  } else if (supabaseAuthState === 'sending') {
JS

text.sub!(old, new)

text.sub!(
  "  } else {\n    status.textContent = config.required\n      ? 'Sign in with Supabase before starting a new analysis.'\n      : 'Optional: sign in with Supabase so new runs show who started them.';\n  }\n",
  "  } else {\n    status.textContent = 'Please enter your email address to verify your identity.';\n  }\n",
)

File.write(path, text)
