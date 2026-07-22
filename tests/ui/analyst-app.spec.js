const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const model = {
  provider: 'openai',
  model: 'gpt-5.4-mini',
  label: 'GPT-5.4 mini',
  summary: 'Balanced analysis model',
  available: true,
  selectable: true,
  pricing_available: true,
  supports_temperature: true,
  supports_reasoning_effort: true,
  reasoning_effort_options: ['low', 'medium', 'high'],
};

const configFixture = {
  llm: model.label,
  default_llm: model,
  available_models: [model],
  pricing_catalog: {},
  phase_model_defaults: Object.fromEntries(
    ['decomposition', 'answering', 'generation', 'evaluation', 'ranking'].map((phase) => [
      phase,
      { provider: model.provider, model: model.model },
    ]),
  ),
  quality_tiers: [],
  premium_phase_options: [],
  vc_investment_strategy: 'Early-stage European B2B software.',
  vc_strategy_source: 'local',
  supabase_auth: { configured: false, required: false },
};

const companySummary = {
  company_lookup_key: 'novabite',
  company_key: 'novabite',
  company_name: 'NovaBite',
  startup_slug: 'novabite',
  company_url: 'https://novabite.example',
  specter_company_id: 'sp-novabite',
  latest_run_at: '2026-07-15T09:30:00Z',
  run_count: 2,
  latest_run: {
    job_id: 'job-complete',
    status: 'done',
    created_at: '2026-07-15T09:30:00Z',
    composite_score: 82,
    bucket: 'priority_review',
  },
};

const companyDetail = {
  ...companySummary,
  runs: [{
    job_id: 'job-complete',
    status: 'done',
    created_at: '2026-07-15T09:30:00Z',
    company_name: 'NovaBite',
    startup_slug: 'novabite',
    composite_score: 82,
    bucket: 'priority_review',
    dimension_scores: [
      { dimension: 'strategy_fit', adjusted_score: 86 },
      { dimension: 'team', adjusted_score: 78 },
      { dimension: 'upside', adjusted_score: 82 },
    ],
    executive_summary: 'Strong thesis fit with credible early commercial evidence.',
    top_pro_1: 'Clear European enterprise wedge.',
    top_pro_1_score: 9,
    top_contra_1: 'Expansion efficiency remains unproven.',
    top_contra_1_score: 6,
  }],
};

const resultsFixture = {
  mode: 'single',
  company_name: 'NovaBite',
  startup_slug: 'novabite',
  company_url: 'https://novabite.example',
  specter_company_id: 'sp-novabite',
  industry: 'B2B software',
  num_documents: 3,
  num_arguments: 6,
  llm: model.label,
  avg_pro: 8.4,
  avg_contra: 4.1,
  total_score: 82,
  pro_arguments: [
    { score: 9, text: 'Clear European enterprise wedge.' },
    { score: 8, text: 'Strong founder-market fit.' },
  ],
  contra_arguments: [{ score: 6, text: 'Expansion efficiency remains unproven.' }],
  ranking_result: {
    rank: 1,
    composite_score: 82,
    bucket: 'priority_review',
    executive_summary: 'Strong thesis fit with credible early commercial evidence.',
    dimension_scores: [
      { dimension: 'strategy_fit', adjusted_score: 86, confidence: 0.9 },
      { dimension: 'team', adjusted_score: 78, confidence: 0.82 },
      { dimension: 'upside', adjusted_score: 82, confidence: 0.78 },
    ],
  },
  summary_rows: [{ company_name: 'NovaBite', evidence_coverage: 0.8, answered_questions: 8, total_questions: 10 }],
  argument_rows: [],
  qa_rows: [],
  team_members: [],
  run_costs: {
    status: 'complete',
    total_usd: 1.24,
    llm_usd: 1.1,
    perplexity_usd: 0.14,
    llm_tokens: { prompt: 12000, completion: 3500, total: 15500 },
    perplexity_search: { requests: 2 },
    by_model: [{ label: model.label, usd: 1.1, pricing_available: true }],
  },
};

const leadgenBatch = {
  intake_id: 'intake-1',
  batch_id: 'Nordics seed pipeline',
  status: 'pending',
  lead_count: 2,
  accepted_count: 2,
  rejected_count: 0,
  created_at: '2026-07-16T08:00:00Z',
};

const leadgenDetail = {
  batch: leadgenBatch,
  leads: [
    {
      lead_id: 'lead-1',
      company_name: 'Northstar',
      domain: 'northstar.example',
      eligible: true,
      approval_status: 'pending',
      leadgen_bucket: 'priority',
      thesis_key: 'b2b-software',
      thesis_status: 'fit',
      context_payload: { source: 'research', rationale: 'European B2B workflow software.' },
    },
    {
      lead_id: 'lead-2',
      company_name: 'Harbor',
      domain: 'harbor.example',
      eligible: true,
      approval_status: 'pending',
      leadgen_bucket: 'watch',
      thesis_key: 'fintech',
      thesis_status: 'review',
      context_payload: { source: 'conference', rationale: 'Early traction requires validation.' },
    },
  ],
};

function json(route, body, status = 200) {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

async function installAppMocks(page, overrides = {}) {
  const state = { requests: [], unexpected: [] };
  const expectedBlockedHosts = new Set(['fonts.googleapis.com', 'fonts.gstatic.com', 'unpkg.com']);

  await page.addInitScript(() => {
    localStorage.setItem('session_id', 'playwright-session');
    localStorage.setItem('app_state_version', '2026-03-15-analysis-state-v2');
    localStorage.setItem('onboarding_welcome_dismissed', '1');
    localStorage.setItem('input_mode_callout_dismissed', '1');
  });

  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const key = `${request.method()} ${url.pathname}`;

    if (url.origin !== 'http://127.0.0.1:4173') {
      if (!expectedBlockedHosts.has(url.hostname)) state.unexpected.push(`${request.method()} ${request.url()}`);
      return route.abort('blockedbyclient');
    }
    if (!url.pathname.startsWith('/api/')) return route.continue();

    let body = null;
    try { body = request.postDataJSON(); } catch (_) { body = request.postData(); }
    state.requests.push({ method: request.method(), path: url.pathname, search: url.search, body });

    if (overrides[key]) return overrides[key](route, request, state);
    if (url.pathname === '/api/public-auth-config') return json(route, { supabase_auth: { configured: false, required: false } });
    if (url.pathname === '/api/check-session') return json(route, { authenticated: true });
    if (url.pathname === '/api/config') return json(route, configFixture);
    if (url.pathname === '/api/web-search-available') return json(route, { available: true });
    if (url.pathname === '/api/jobs') return json(route, { jobs: [] });
    if (url.pathname === '/api/company-runs/summary') return json(route, { companies: [companySummary], total: 1, next_offset: null });
    if (url.pathname === '/api/company-runs/detail/novabite') {
      return json(route, {
        company: {
          ...companyDetail,
          runs: companyDetail.runs.map((run) => ({ ...run, results: resultsFixture })),
        },
      });
    }
    if (url.pathname === '/api/companies/novabite/chat') {
      return json(route, {
        transcript: [],
        source_counts: { runs: 2 },
        run_count: 2,
        available_models: [model],
        llm_provider: model.provider,
        llm_model: model.model,
        model_label: model.label,
      });
    }
    if (url.pathname === '/api/leadgen/intakes') return json(route, { batches: [leadgenBatch] });
    if (url.pathname === '/api/leadgen/intakes/intake-1') return json(route, leadgenDetail);
    if (url.pathname === '/api/costs/job-complete') return json(route, { stages: [] });
    if (url.pathname === '/api/feedback') return json(route, { id: 'feedback-1' }, 201);
    if (url.pathname === '/api/settings/vc-strategy') return json(route, { vc_investment_strategy: 'Updated strategy', vc_strategy_source: 'supabase' });

    state.unexpected.push(`${request.method()} ${url.pathname}${url.search}`);
    return json(route, { detail: `Unexpected mocked request: ${request.method()} ${url.pathname}` }, 599);
  });

  return state;
}

async function openAuthenticatedApp(page, overrides = {}) {
  const browserErrors = [];
  page.on('pageerror', (error) => browserErrors.push(error.message));
  page.on('console', (message) => {
    if (message.type() !== 'error') return;
    const text = message.text();
    if (/Failed to load resource: net::ERR_(FAILED|BLOCKED_BY_CLIENT)/.test(text)) return;
    browserErrors.push(`console: ${text}`);
  });
  const state = await installAppMocks(page, overrides);
  await page.goto('/');
  await expect(page.locator('#upload-screen')).toHaveClass(/active/);
  return { ...state, browserErrors };
}

test('authenticated shell keeps every primary workflow reachable @webkit', async ({ page }) => {
  const state = await openAuthenticatedApp(page);

  await expect(page.getByRole('heading', { name: 'Rank your deal flow.' })).toBeVisible();
  await page.locator('#analysis-open-upload').click();
  await expect(page.locator('#analysis-screen')).toHaveClass(/active/);
  await page.locator('#saved-jobs-open-analysis').click();
  await expect(page.locator('#companies-screen')).toHaveClass(/active/);
  await expect(page.getByText('NovaBite', { exact: true }).first()).toBeVisible();
  await page.locator('#leadgen-open-companies').click();
  await expect(page.locator('#leadgen-screen')).toHaveClass(/active/);
  await expect(page.getByRole('heading', { name: 'Nordics seed pipeline' })).toBeVisible();
  await page.locator('#new-analysis-btn-leadgen').click();
  await expect(page.locator('#upload-screen')).toHaveClass(/active/);

  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('password failure and success preserve the login and session contract', async ({ page }) => {
  const submittedPasswords = [];
  const state = await openAuthenticatedApp(page, {
    'POST /api/login': (route, request) => {
      const body = request.postDataJSON();
      submittedPasswords.push(body.password);
      if (body.password === 'wrong') return json(route, { detail: 'Wrong password' }, 401);
      return json(route, { session_id: 'new-playwright-session' });
    },
  });

  await page.evaluate(() => {
    sessionId = null;
    localStorage.removeItem('session_id');
    showScreen('password-screen');
    document.querySelector('#login-email-stage').hidden = true;
    document.querySelector('#login-unlock-stage').hidden = false;
  });
  await page.locator('#password-input').fill('wrong');
  await page.locator('#login-unlock-button').click();
  await expect(page.locator('#password-error')).toHaveText('Wrong password. Try again.');

  await page.locator('#password-input').fill('correct');
  await page.locator('#login-unlock-button').click();
  await expect(page.locator('#upload-screen')).toHaveClass(/active/);
  await expect.poll(() => page.evaluate(() => localStorage.getItem('session_id')))
    .toBe('new-playwright-session');
  expect(submittedPasswords).toEqual(['wrong', 'correct']);
  expect(state.browserErrors).toEqual([
    'console: Failed to load resource: the server responded with a status of 401 (Unauthorized)',
  ]);
  expect(state.unexpected).toEqual([]);
});

test('file selection, drag and drop, and all intake modes stay intact', async ({ page }) => {
  const state = await openAuthenticatedApp(page);

  await page.locator('#file-input').setInputFiles([
    { name: 'company_export.csv', mimeType: 'text/csv', buffer: Buffer.from('company_name,url\nNovaBite,novabite.example') },
    { name: 'people.csv', mimeType: 'text/csv', buffer: Buffer.from('name,company\nAda,NovaBite') },
  ]);
  await expect(page.getByText('company_export.csv', { exact: true })).toBeVisible();
  await expect(page.getByText('people.csv', { exact: true })).toBeVisible();
  await expect(page.locator('#specter-badge')).toBeVisible();

  await page.locator('[data-mode="original"]').click();
  await expect(page.locator('#input-mode-hint')).toContainText('All files = one company');
  await page.locator('[data-mode="pitchdeck"]').click();
  await expect(page.locator('#specter-mcp-toggle')).toBeVisible();

  const dataTransfer = await page.evaluateHandle(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(['pitch deck'], 'novabite-deck.pdf', { type: 'application/pdf' }));
    return transfer;
  });
  await page.locator('#upload-zone').dispatchEvent('drop', { dataTransfer });
  await expect(page.getByText('novabite-deck.pdf', { exact: true })).toBeVisible();
  await page.locator('[data-mode="specter"]').click();
  await expect(page.locator('#specter-urls-section')).toBeVisible();
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('analysis intake preserves controls and request payloads', async ({ page }) => {
  let uploadPayload = null;
  let analyzePayload = null;
  const state = await openAuthenticatedApp(page, {
    'POST /api/upload-urls': (route, request) => {
      uploadPayload = request.postDataJSON();
      return json(route, { job_id: 'job-new' });
    },
    'POST /api/analyze/job-new': (route, request) => {
      analyzePayload = request.postDataJSON();
      return json(route, { status: 'running' });
    },
  });

  await page.locator('#specter-urls-toggle').click();
  await page.locator('#specter-urls-input').fill('novabite.example');
  await page.locator('#run-name-input').fill('UI contract run');
  await page.locator('#instructions-input').fill('Focus on European B2B fit.');
  await page.locator('#web-search-toggle .toggle-row').click();
  await page.locator('[data-web-intensity="targeted"]').click();
  await page.locator('#analyze-btn').click();

  await expect.poll(() => uploadPayload).not.toBeNull();
  await expect.poll(() => analyzePayload).not.toBeNull();
  expect(uploadPayload.urls).toEqual(['novabite.example']);
  expect(analyzePayload.run_name).toBe('UI contract run');
  expect(analyzePayload.input_mode).toBe('specter');
  expect(analyzePayload.instructions).toBe('Focus on European B2B fit.');
  expect(analyzePayload.use_web_search).toBe(true);
  expect(analyzePayload.web_search_mode).toBe('targeted');
  expect(analyzePayload.phase_models).toBeTruthy();
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('results retain costs, evidence, links, and audit interactions', async ({ page }) => {
  const state = await openAuthenticatedApp(page);

  await page.evaluate((fixture) => {
    stageCostsJobId = 'job-complete';
    renderResults(fixture);
    showScreen('results-screen');
  }, resultsFixture);

  await expect(page.locator('#results-screen')).toHaveClass(/active/);
  await expect(page.getByText('Total Run Cost')).toBeVisible();
  await expect(page.getByText('Evidence coverage: 80%')).toBeVisible();
  await expect(page.getByRole('link', { name: 'novabite.example' })).toHaveAttribute('target', '_blank');
  await page.getByRole('button', { name: 'View log' }).click();
  await expect(page.locator('#audit-modal-overlay')).toHaveClass(/audit-modal-open/);
  await page.keyboard.press('Escape');
  await expect(page.locator('#audit-modal-overlay')).not.toHaveClass(/audit-modal-open/);
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('LeadGen approval and feedback retain their API contracts', async ({ page }) => {
  let approvalPayload = null;
  let rejectionPayload = null;
  let settingsPayload = null;
  let feedbackPayload = null;
  const state = await openAuthenticatedApp(page, {
    'POST /api/leadgen/intakes/intake-1/approve': (route, request) => {
      approvalPayload = request.postDataJSON();
      return json(route, { job_id: 'leadgen-job', status: 'queued' });
    },
    'POST /api/leadgen/intakes/intake-1/reject': (route, request) => {
      rejectionPayload = request.postDataJSON();
      return json(route, { status: 'rejected' });
    },
    'POST /api/settings/vc-strategy': (route, request) => {
      settingsPayload = request.postDataJSON();
      return json(route, { vc_investment_strategy: settingsPayload.vc_investment_strategy, vc_strategy_source: 'supabase' });
    },
    'POST /api/feedback': (route, request) => {
      feedbackPayload = request.postDataJSON();
      return json(route, { id: 'feedback-1' }, 201);
    },
  });

  await page.locator('#leadgen-open-upload').click();
  await page.getByRole('button', { name: /Nordics seed pipeline/ }).click();
  await page.locator('[data-leadgen-lead="lead-1"]').check();
  await page.locator('[data-leadgen-approve-selected]').click();
  await expect.poll(() => approvalPayload).not.toBeNull();
  expect(approvalPayload.lead_ids).toEqual(['lead-1']);

  await page.locator('[data-leadgen-lead="lead-2"]').check();
  await page.locator('[data-leadgen-reject-selected]').click();
  await expect.poll(() => rejectionPayload).not.toBeNull();
  expect(rejectionPayload.lead_ids).toEqual(['lead-2']);

  await page.locator('#new-analysis-btn-leadgen').click();
  await page.evaluate(() => { analysisConfig.vc_strategy_source = 'supabase'; });
  await page.locator('#settings-open').click();
  await expect(page.locator('#settings-modal-overlay')).toHaveClass(/audit-modal-open/);
  await page.locator('#vc-strategy-input').fill('Updated European B2B strategy.');
  await page.locator('#settings-save-vc-btn').click();
  await expect.poll(() => settingsPayload).not.toBeNull();
  expect(settingsPayload).toEqual({ vc_investment_strategy: 'Updated European B2B strategy.' });
  await expect.poll(() => page.evaluate(() => localStorage.getItem('vc_investment_strategy')))
    .toBe('Updated European B2B strategy.');
  await page.evaluate(() => {
    supabaseIdentity = { access_token: 'playwright-token', label: 'Playwright user' };
    syncFeedbackButtonVisibility();
  });
  await page.locator('#feedback-bubble').click();
  await page.locator('.hero-text h1').click();
  await page.locator('#feedback-comment').fill('The heading needs a visual check.');
  await page.locator('#feedback-submit-btn').click();
  await expect.poll(() => feedbackPayload).not.toBeNull();
  expect(feedbackPayload.comment).toBe('The heading needs a visual check.');
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('company search, collapse, detail, and chat preserve their contracts', async ({ page }) => {
  let chatPayload = null;
  const state = await openAuthenticatedApp(page, {
    'POST /api/companies/novabite/chat': (route, request) => {
      chatPayload = request.postDataJSON();
      return json(route, {
        transcript: [
          { role: 'user', content: chatPayload.message, citations: [] },
          { role: 'assistant', content: 'NovaBite has a credible enterprise wedge.', citations: [] },
        ],
        source_counts: { runs: 2 },
        run_count: 2,
        available_models: [model],
        llm_provider: model.provider,
        llm_model: model.model,
        model_label: model.label,
      });
    },
  });

  await page.locator('#saved-jobs-open-upload').click();
  await page.locator('[data-company-select="novabite"]').click();
  await expect(page.getByRole('heading', { name: 'Chat · NovaBite' })).toBeVisible();

  await page.locator('[data-company-chat-input="novabite"]').fill('What is the strongest evidence?');
  await page.locator('[data-company-chat-input="novabite"]').press('Enter');
  await expect.poll(() => chatPayload).not.toBeNull();
  expect(chatPayload).toMatchObject({
    message: 'What is the strongest evidence?',
    active_job_id: 'job-complete',
    llm_provider: model.provider,
    llm_model: model.model,
  });
  await expect(page.getByText('NovaBite has a credible enterprise wedge.')).toBeVisible();

  await page.locator('[data-companies-sidebar-toggle="collapse"]').click();
  await expect(page.locator('[data-companies-sidebar-toggle="expand"]')).toBeVisible();
  await page.locator('[data-companies-sidebar-toggle="expand"]').click();
  await page.locator('#companies-search-input').fill('missing');
  await expect(page.locator('[data-company-select]')).toHaveCount(0);
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('desktop shell has no serious accessibility violations', async ({ page }) => {
  const state = await openAuthenticatedApp(page);
  const scan = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  const serious = scan.violations.filter((violation) => ['serious', 'critical'].includes(violation.impact));

  expect(serious).toEqual([]);
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('processing state keeps progress and stop controls usable', async ({ page }) => {
  let controlPayload = null;
  const state = await openAuthenticatedApp(page, {
    'POST /api/jobs/job-running/control': (route, request) => {
      controlPayload = request.postDataJSON();
      return json(route, { status: 'stopping' });
    },
  });

  await page.evaluate(() => {
    currentJobId = 'job-running';
    showScreen('processing-screen');
    document.querySelector('#processing-status').textContent = 'Generating arguments for NovaBite';
    renderProgressFeed([
      'Analyzing NovaBite (1/2) — [1/4] Running evidence retrieval',
      'Analyzing NovaBite (1/2) — [3/4] Generating arguments (6/10 Q&A)',
    ], 'Generating arguments for NovaBite', 'running');
    setProcessingControls('running');
  });

  await expect(page.locator('#processing-company-meta')).toContainText('Company 1 of 2');
  await expect(page.locator('#progress-bar-pct')).not.toBeEmpty();
  await page.locator('#stop-btn').click();
  await expect.poll(() => controlPayload).not.toBeNull();
  expect(controlPayload).toEqual({ action: 'stop' });
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('keyboard focus and modal dismissal remain visible and operable', async ({ page }) => {
  const state = await openAuthenticatedApp(page);

  await page.keyboard.press('Tab');
  const focused = page.locator(':focus');
  await expect(focused).toBeVisible();
  const outlineStyle = await focused.evaluate((element) => getComputedStyle(element).outlineStyle);
  expect(outlineStyle).not.toBe('none');

  await page.locator('#settings-open').click();
  await expect(page.locator('#settings-modal-overlay')).toHaveClass(/audit-modal-open/);
  await page.keyboard.press('Escape');
  await expect(page.locator('#settings-modal-overlay')).not.toHaveClass(/audit-modal-open/);
  expect(state.browserErrors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

for (const viewport of [
  { name: 'tablet', width: 820, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test(`${viewport.name} layout has no page overflow or hidden primary actions`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    const state = await openAuthenticatedApp(page);

    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
    await expect(page.locator('#analyze-btn')).toBeAttached();

    await page.locator('#settings-open').click();
    const dialogBox = await page.locator('#settings-modal-overlay .audit-modal-panel').boundingBox();
    expect(dialogBox.x).toBeGreaterThanOrEqual(0);
    expect(dialogBox.y).toBeGreaterThanOrEqual(0);
    expect(dialogBox.x + dialogBox.width).toBeLessThanOrEqual(viewport.width + 1);
    expect(dialogBox.y + dialogBox.height).toBeLessThanOrEqual(viewport.height + 1);
    expect(state.browserErrors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
}

test.describe('Matchbook visual baselines', () => {
  test.skip(process.platform !== 'linux', 'Visual baselines are authoritative in Linux Chromium CI.');

  test('login, intake, processing, results, companies, and LeadGen', async ({ page }) => {
    const state = await openAuthenticatedApp(page);

    await page.evaluate(() => {
      supabaseIdentity = {
        access_token: 'visual-token',
        email: 'analyst@rockaway.vc',
        label: 'analyst@rockaway.vc',
      };
      renderLoginIdentity();
      showScreen('password-screen');
    });
    await expect(page).toHaveScreenshot('login-desktop.png');

    await page.evaluate(() => openNewAnalysis());
    await expect(page).toHaveScreenshot('intake-desktop.png');

    await page.locator('#analysis-open-upload').click();
    await expect(page.locator('#analysis-screen')).toHaveClass(/active/);
    await expect(page).toHaveScreenshot('analysis-history-empty-desktop.png');
    await page.locator('#new-analysis-btn-history').click();

    await page.evaluate(() => {
      showScreen('upload-screen');
      showSettingsModal();
    });
    await expect(page.locator('#settings-modal-overlay')).toBeVisible();
    await expect(page).toHaveScreenshot('settings-modal-desktop.png');
    await page.evaluate(() => hideSettingsModal());

    await page.evaluate(() => {
      showScreen('processing-screen');
      document.querySelector('#processing-status').textContent = 'Generating arguments for NovaBite';
      renderProgressFeed([
        'Analyzing NovaBite (1/2) — [1/4] Running evidence retrieval',
        'Analyzing NovaBite (1/2) — [3/4] Generating arguments (6/10 Q&A)',
      ], 'Generating arguments for NovaBite', 'running');
    });
    await expect(page).toHaveScreenshot('processing-desktop.png');

    await page.evaluate((fixture) => {
      stageCostsJobId = 'job-complete';
      renderResults(fixture);
      showScreen('results-screen');
    }, resultsFixture);
    await expect(page).toHaveScreenshot('results-desktop.png');
    await page.getByRole('button', { name: 'View log' }).click();
    await expect(page.locator('#audit-modal-overlay')).toHaveClass(/audit-modal-open/);
    await expect(page).toHaveScreenshot('audit-modal-desktop.png');
    await page.keyboard.press('Escape');

    await page.locator('#saved-jobs-open-results').click();
    await page.evaluate(({ summary, detail, results }) => {
      savedCompaniesCache = [summary];
      savedCompanyDetailsCache = {
        novabite: {
          ...detail,
          runs: detail.runs.map((run) => ({ ...run, results })),
        },
      };
      savedCompanyDetailLoadState = { novabite: { loading: false, loaded: true, error: '' } };
      selectedCompanyKey = 'novabite';
      selectedCompanyJobId = 'job-complete';
      renderSavedCompaniesWindow();
    }, { summary: companySummary, detail: companyDetail, results: resultsFixture });
    await expect(page.getByText('NovaBite', { exact: true }).first()).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Chat · NovaBite' })).toBeVisible();
    await expect(page).toHaveScreenshot('companies-desktop.png');

    await page.locator('#leadgen-open-companies').click();
    await expect(page.getByRole('heading', { name: 'Nordics seed pipeline' })).toBeVisible();
    await expect(page).toHaveScreenshot('leadgen-desktop.png');
    expect(state.browserErrors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });

  test('mobile intake', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    const state = await openAuthenticatedApp(page);
    await expect(page).toHaveScreenshot('intake-mobile.png');
    expect(state.browserErrors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
});
