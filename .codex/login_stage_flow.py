from pathlib import Path

path = Path("/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/web/static/index.html")
text = path.read_text()

old_markup = """    <div class="password-card">
      <div class="logo">
        <div class="logo-icon">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="3" y="12" width="4" height="8" rx="1.2"></rect>
            <rect x="10" y="7" width="4" height="13" rx="1.2"></rect>
            <rect x="17" y="3" width="4" height="17" rx="1.2"></rect>
          </svg>
        </div>
        <div class="logo-lockup">
          <div class="logo-name">Rockaway</div>
          <div class="logo-subtitle">Deal Intelligence</div>
        </div>
      </div>
      <div class="password-meta">
        <span class="password-kicker">Secure access</span>
        <h2>Enter workspace</h2>
        <p>Protected internal interface for reviewing pipelines, company runs, and portfolio screening output.</p>
      </div>
      <form id="login-form">
        <label class="password-field-label" for="login-email-input">Email</label>
        <input type="email" class="password-input" id="login-email-input" placeholder="name@rockaway.vc" maxlength="200" autocomplete="email" autofocus>
        <div style="display:flex;gap:12px;align-items:center;margin:12px 0 22px;flex-wrap:wrap;">
          <button type="button" class="btn-ghost" id="login-send-code">Request verification code</button>
          <div class="password-error" id="login-identity-status" style="margin:0;"></div>
        </div>
        <label class="password-field-label" for="login-code-input">Verification code</label>
        <input type="text" class="password-input" id="login-code-input" placeholder="Enter code from email" maxlength="12" autocomplete="one-time-code" inputmode="numeric">
        <label class="password-field-label" for="password-input">Password</label>
        <input type="password" class="password-input" id="password-input" placeholder="Enter password" maxlength="20">
        <div class="password-error" id="password-error"></div>
        <button type="submit" class="btn-green">Unlock</button>
      </form>
      <div class="password-footnote">This session stays local to your browser and is validated against the server before any analysis routes are opened.</div>
    </div>"""

new_markup = """    <div class="password-card">
      <div class="password-meta">
        <span class="password-kicker">Secure access</span>
        <h2>Enter workspace</h2>
        <p>Protected internal interface for reviewing pipelines, company runs, and portfolio screening output.</p>
      </div>
      <form id="login-form">
        <label class="password-field-label" for="login-email-input">Email</label>
        <input type="email" class="password-input" id="login-email-input" placeholder="name@rockaway.vc" maxlength="200" autocomplete="email" autofocus>
        <div style="display:flex;gap:12px;align-items:flex-start;margin:12px 0 22px;flex-direction:column;">
          <button type="button" class="btn-green" id="login-send-code">Request verification code</button>
          <div class="password-error" id="login-identity-status" style="margin:0;"></div>
        </div>
        <div id="login-verification-stage" style="display:none;">
          <label class="password-field-label" for="login-code-input">Verification code</label>
          <input type="text" class="password-input" id="login-code-input" placeholder="Enter code from email" maxlength="12" autocomplete="one-time-code" inputmode="numeric">
          <label class="password-field-label" for="password-input">Password</label>
          <input type="password" class="password-input" id="password-input" placeholder="Enter password" maxlength="20">
          <div class="password-error" id="password-error"></div>
          <button type="submit" class="btn-green">Unlock</button>
        </div>
      </form>
      <div class="password-footnote">This session stays local to your browser and is validated against the server before any analysis routes are opened.</div>
    </div>"""

text = text.replace(old_markup, new_markup)

text = text.replace(
    "let supabaseAuthError = '';\n",
    "let supabaseAuthError = '';\nlet loginVerificationRequested = false;\n",
)

old_render = """function renderLoginIdentity() {
  const status = $('#login-identity-status');
  const emailInput = $('#login-email-input');
  const codeInput = $('#login-code-input');
  const sendBtn = $('#login-send-code');
  if (!status || !emailInput || !codeInput || !sendBtn) return;

  const config = getSupabaseAuthConfig();
  const authReady = !!getSupabaseAccessToken();
  const label = supabaseIdentity?.label || '';
  const email = supabaseIdentity?.email || emailInput.value.trim() || supabaseMagicLinkEmail || '';

  sendBtn.disabled = supabaseAuthState === 'sending';

  if (authReady) {
    status.textContent = email && label && label !== email
      ? `Email verified as ${label} · ${email}. Enter your workspace password to continue.`
      : `Email verified as ${label || email || 'this user'}. Enter your workspace password to continue.`;
  } else if (!config?.url) {
    status.textContent = '';
  } else if (!config?.configured) {
    status.textContent = 'Email verification is not configured yet in this local environment.';
  } else if (supabaseAuthState === 'sending') {
    status.textContent = 'Sending verification code…';
  } else if (supabaseAuthState === 'pending') {
    status.textContent = supabaseMagicLinkEmail
      ? `Verification email sent to ${supabaseMagicLinkEmail}. Enter the code from the email, then your password.`
      : 'Verification email sent. Enter the code from the email, then your password.';
  } else if (supabaseAuthState === 'error' && supabaseAuthError) {
    status.textContent = supabaseAuthError;
  } else {
    status.textContent = 'Enter your email address, request a verification code, then enter the code and your password.';
  }
}
"""

new_render = """function renderLoginIdentity() {
  const status = $('#login-identity-status');
  const emailInput = $('#login-email-input');
  const codeInput = $('#login-code-input');
  const sendBtn = $('#login-send-code');
  const stage = $('#login-verification-stage');
  if (!status || !emailInput || !codeInput || !sendBtn || !stage) return;

  const config = getSupabaseAuthConfig();
  const authReady = !!getSupabaseAccessToken();
  const label = supabaseIdentity?.label || '';
  const email = supabaseIdentity?.email || emailInput.value.trim() || supabaseMagicLinkEmail || '';
  const stageVisible = loginVerificationRequested || supabaseAuthState === 'pending' || supabaseAuthState === 'error' || authReady;

  sendBtn.disabled = supabaseAuthState === 'sending';
  stage.style.display = stageVisible ? 'block' : 'none';

  if (authReady) {
    status.textContent = email && label && label !== email
      ? `Email verified as ${label} · ${email}. Enter your workspace password to continue.`
      : `Email verified as ${label || email || 'this user'}. Enter your workspace password to continue.`;
  } else if (!config?.url) {
    status.textContent = '';
  } else if (!stageVisible) {
    status.textContent = 'Please enter your email address to verify your identity.';
  } else if (!config?.configured) {
    status.textContent = 'Email verification is not configured yet in this local environment.';
  } else if (supabaseAuthState === 'sending') {
    status.textContent = 'Sending verification code…';
  } else if (supabaseAuthState === 'pending') {
    status.textContent = supabaseMagicLinkEmail
      ? `Verification email sent to ${supabaseMagicLinkEmail}. Enter the code from the email, then your password.`
      : 'Verification email sent. Enter the code from the email, then your password.';
  } else if (supabaseAuthState === 'error' && supabaseAuthError) {
    status.textContent = supabaseAuthError;
  } else {
    status.textContent = 'Please enter your email address to verify your identity.';
  }
}
"""

text = text.replace(old_render, new_render)

text = text.replace(
    """$('#login-send-code')?.addEventListener('click', async () => {
  const email = ($('#login-email-input')?.value || '').trim();
  if (!email) {
    supabaseAuthState = 'error';
    supabaseAuthError = 'Enter your email address to receive a verification code.';
    renderLoginIdentity();
    return;
  }
  try {
    supabaseAuthState = 'sending';
    supabaseMagicLinkEmail = email;
    supabaseAuthError = '';
    renderLoginIdentity();
""",
    """$('#login-send-code')?.addEventListener('click', async () => {
  const email = ($('#login-email-input')?.value || '').trim();
  if (!email) {
    supabaseAuthState = 'error';
    supabaseAuthError = 'Enter your email address to receive a verification code.';
    renderLoginIdentity();
    return;
  }
  try {
    loginVerificationRequested = true;
    supabaseAuthState = 'sending';
    supabaseMagicLinkEmail = email;
    supabaseAuthError = '';
    renderLoginIdentity();
""",
)

text = text.replace(
    """    if (/verification|email|code|supabase/i.test(msg)) {
      supabaseAuthState = 'error';
      supabaseAuthError = msg;
      renderLoginIdentity();
      $('#login-code-input')?.focus();
    } else {
""",
    """    if (/verification|email|code|supabase/i.test(msg)) {
      loginVerificationRequested = true;
      supabaseAuthState = 'error';
      supabaseAuthError = msg;
      renderLoginIdentity();
      $('#login-code-input')?.focus();
    } else {
""",
)

path.write_text(text)
