from pathlib import Path

path = Path("/Users/dusan.zabrodsky/Library/CloudStorage/OneDrive-Personal/Rockaway/Ventures/Deal Intelligence/.codex/worktrees/review-supabase-auth/web/static/index.html")
text = path.read_text()

old = """// Check existing session on load
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

new = """// Check existing session on load
(async () => {
  await loadPublicAuthConfig();
  await initializeSupabaseAuth();
  renderLoginIdentity();

  const requiresSupabase = isSupabaseAuthRequired();
  const hasSupabaseIdentity = !!getSupabaseAccessToken();
  if (requiresSupabase && !hasSupabaseIdentity) {
    sessionId = null;
    localStorage.removeItem('session_id');
    setCookie('session_id', '', 0);
    syncRoute();
    return;
  }

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

if old not in text:
    raise SystemExit("startup block not found")

path.write_text(text.replace(old, new))
