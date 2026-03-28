from pathlib import Path
import re

ROOT = Path("/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence")
STATIC = ROOT / ".codex/worktrees/review-supabase-auth/web/static/index.html"
APP = ROOT / ".codex/worktrees/review-supabase-auth/web/app.py"


def patch_static() -> None:
    text = STATIC.read_text()

    text = text.replace(
        """      <form id="login-form">
        <label class="password-field-label" for="password-input">Password</label>
        <input type="password" class="password-input" id="password-input" placeholder="Enter password" maxlength="20" autofocus>
        <div class="password-error" id="password-error"></div>
        <button type="submit" class="btn-green">Unlock</button>
      </form>""",
        """      <form id="login-form">
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
      </form>""",
    )

    text = text.replace(
        "let analysisConfig = null;\n",
        "let analysisConfig = null;\nlet publicSupabaseAuthConfig = null;\n",
    )

    text = text.replace(
        """function getSupabaseAuthConfig() {
  return analysisConfig?.supabase_auth || null;
}
""",
        """function getSupabaseAuthConfig() {
  return analysisConfig?.supabase_auth || publicSupabaseAuthConfig || null;
}

async function loadPublicAuthConfig() {
  try {
    const response = await fetch('/api/public-auth-config', { credentials: 'include' });
    const data = await response.json().catch(() => ({}));
    publicSupabaseAuthConfig = data?.supabase_auth || null;
  } catch (_) {
    publicSupabaseAuthConfig = null;
  }
  return publicSupabaseAuthConfig;
}
""",
    )

    render_block_pattern = re.compile(
        r"function renderAnalysisIdentity\(\) \{.*?\n\}\n\nfunction updateAnalyzeButtonState\(\) \{",
        re.S,
    )
    render_block_replacement = """function renderLoginIdentity() {
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

function renderAnalysisIdentity() {
  const section = $('#analysis-identity-section');
  if (section) {
    section.style.display = 'none';
  }
}

function updateAnalyzeButtonState() {"""
    text = render_block_pattern.sub(render_block_replacement, text, count=1)

    text = text.replace(
        """function setSupabaseIdentityFromSession(session) {
  const user = session?.user || null;
  const metadata = user?.user_metadata || {};
  const displayName = String(
    metadata?.full_name || metadata?.name || metadata?.display_name || metadata?.user_name || ''
  ).trim();
  const email = String(user?.email || '').trim();
  supabaseIdentity = session && user ? {
    user_id: user.id || null,
    email: email || null,
    display_name: displayName || null,
    label: displayName || email || null,
    access_token: session.access_token || null,
  } : null;
}
""",
        """function setSupabaseIdentityFromSession(session) {
  const user = session?.user || null;
  const metadata = user?.user_metadata || {};
  const displayName = String(
    metadata?.full_name || metadata?.name || metadata?.display_name || metadata?.user_name || ''
  ).trim();
  const email = String(user?.email || '').trim();
  supabaseIdentity = session && user ? {
    user_id: user.id || null,
    email: email || null,
    display_name: displayName || null,
    label: displayName || email || null,
    access_token: session.access_token || null,
  } : null;
  if (supabaseIdentity?.email) {
    const emailField = $('#login-email-input');
    if (emailField) emailField.value = supabaseIdentity.email;
  }
}
""",
    )

    text = text.replace(
        """      renderAnalysisIdentity();
      updateAnalyzeButtonState();
    });
    supabaseAuthListenerBound = true;
  }
""",
        """      renderLoginIdentity();
      renderAnalysisIdentity();
      updateAnalyzeButtonState();
    });
    supabaseAuthListenerBound = true;
  }
""",
    )

    text = text.replace(
        """  } catch (_) {
    supabaseAuthState = 'signed_out';
    supabaseIdentity = null;
  }
  return supabaseIdentity;
}
""",
        """  } catch (_) {
    supabaseAuthState = 'signed_out';
    supabaseIdentity = null;
  }
  renderLoginIdentity();
  return supabaseIdentity;
}
""",
    )

    password_block_pattern = re.compile(
        r"// ── Password ──\n.*?\n\nasync function checkWebSearchAvailable\(\) \{",
        re.S,
    )
    password_block_replacement = """// ── Password ──
$('#login-send-code')?.addEventListener('click', async () => {
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
    const client = await ensureSupabaseClient();
    if (!client) {
      throw new Error('Email verification is not configured yet in this local environment.');
    }
    const { error } = await client.auth.signInWithOtp({
      email,
      options: {
        shouldCreateUser: false,
        emailRedirectTo: getSupabaseAuthConfig()?.redirect_url || window.location.href,
      },
    });
    if (error) throw error;
    supabaseAuthState = 'pending';
    supabaseAuthError = '';
    renderLoginIdentity();
    $('#login-code-input')?.focus();
  } catch (err) {
    supabaseAuthState = 'error';
    supabaseAuthError = err?.message || 'Failed to send verification code.';
    renderLoginIdentity();
  }
});

$('#login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const pw = $('#password-input').value;
  const email = ($('#login-email-input')?.value || '').trim();
  const code = ($('#login-code-input')?.value || '').trim();
  $('#password-error').textContent = '';

  try {
    const config = getSupabaseAuthConfig();
    if (config?.required && !getSupabaseAccessToken()) {
      if (!email) {
        throw new Error('Enter your email address first.');
      }
      if (!code) {
        throw new Error('Enter the verification code from your email.');
      }
      const client = await ensureSupabaseClient();
      if (!client) {
        throw new Error('Email verification is not configured yet in this local environment.');
      }
      const { data, error } = await client.auth.verifyOtp({
        email,
        token: code,
        type: 'email',
      });
      if (error) throw error;
      setSupabaseIdentityFromSession(data?.session || null);
      supabaseAuthState = 'signed_in';
      supabaseAuthError = '';
      renderLoginIdentity();
      if (!getSupabaseAccessToken()) {
        throw new Error('Email verification did not complete. Request a new code and try again.');
      }
    }

    const data = await api('/api/login', {
      method: 'POST',
      body: JSON.stringify({
        password: pw,
        supabase_access_token: getSupabaseAccessToken(),
      }),
    });
    sessionId = data.session_id;
    localStorage.setItem('session_id', sessionId);
    setCookie('session_id', sessionId);
    syncRoute();
    checkWebSearchAvailable();
    await loadAnalysisConfig();
    renderRecentRuns();
    initOnboardingWelcome();
    initInputModeCallout();
    backfillRecentRunsWithLlm();
  } catch (err) {
    const msg = err.message === 'Wrong password' ? 'Wrong password. Try again.'
      : err.message || 'Connection error. Is the server running?';
    if (/verification|email|code|supabase/i.test(msg)) {
      supabaseAuthState = 'error';
      supabaseAuthError = msg;
      renderLoginIdentity();
      $('#login-code-input')?.focus();
    } else {
      $('#password-error').textContent = msg;
      $('#password-input').value = '';
      $('#password-input').focus();
    }
  }
});

async function checkWebSearchAvailable() {"""
    text = password_block_pattern.sub(password_block_replacement, text, count=1)

    startup_pattern = re.compile(
        r"// Check existing session on load\n\(async \(\) => \{\n.*?\n\}\)\(\);\n",
        re.S,
    )
    startup_replacement = """// Check existing session on load
(async () => {
  await loadPublicAuthConfig();
  await initializeSupabaseAuth();
  renderLoginIdentity();

  if (sessionId) {
    setCookie('session_id', sessionId);
    try {
      const data = await api('/api/check-session');
      if (data.authenticated) {
        checkWebSearchAvailable();
        await loadAnalysisConfig();
        await initializeSupabaseAuth();
        renderRecentRuns();
        renderSavedCompaniesWindow();
        await fetchSavedJobs();
        fetchSavedCompanies();
        backfillRecentRunsWithLlm();
        const resumed = await resumeActiveJob();
        if (!resumed) {
          openNewAnalysis();
        } else {
          fetchSavedCompanies();
        }
        return;
      }
    } catch {}
  }
  syncRoute();
})();
"""
    text = startup_pattern.sub(startup_replacement, text, count=1)

    STATIC.write_text(text)


def patch_app() -> None:
    text = APP.read_text()

    text = text.replace(
        "class LoginRequest(BaseModel):\n    password: str\n",
        "class LoginRequest(BaseModel):\n    password: str\n    supabase_access_token: str | None = None\n",
    )

    text = text.replace(
        """@app.post("/api/login")
async def login(req: LoginRequest):
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password")
    raw_id = secrets.token_urlsafe(32)
""",
        """@app.post("/api/login")
async def login(req: LoginRequest):
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password")
    config = _supabase_public_auth_config()
    if config["required"]:
        token = (req.supabase_access_token or "").strip()
        if not token:
            raise HTTPException(status_code=401, detail="Email verification required before unlocking.")
        if not callable(getattr(db, "is_configured", None)) or not db.is_configured():
            raise HTTPException(status_code=503, detail="Supabase storage is not configured.")
        get_user = getattr(db, "get_authenticated_supabase_user", None)
        if not callable(get_user):
            raise HTTPException(status_code=503, detail="Supabase auth validation is unavailable.")
        user = await asyncio.to_thread(get_user, token)
        identity = _build_started_by_identity(user)
        if not identity.get("started_by_user_id") or not identity.get("started_by_label"):
            raise HTTPException(status_code=401, detail="Email verification code is invalid or expired.")
    raw_id = secrets.token_urlsafe(32)
""",
    )

    text = text.replace(
        """@app.get("/api/check-session")
async def check_session(session_id: str | None = Cookie(default=None)):
    return {"authenticated": _check_session(session_id)}
""",
        """@app.get("/api/public-auth-config")
async def public_auth_config():
    return {"supabase_auth": _supabase_public_auth_config()}


@app.get("/api/check-session")
async def check_session(session_id: str | None = Cookie(default=None)):
    return {"authenticated": _check_session(session_id)}
""",
    )

    APP.write_text(text)


patch_static()
patch_app()
