const { defineConfig, devices } = require('@playwright/test');

const port = Number(process.env.PLAYWRIGHT_PORT || 4173);
const baseURL = `http://127.0.0.1:${port}`;

module.exports = defineConfig({
  testDir: './tests/ui',
  outputDir: './test-results/playwright',
  snapshotPathTemplate: '{testDir}/__snapshots__/{testFilePath}/{arg}-{projectName}{ext}',
  timeout: 30_000,
  expect: {
    timeout: 7_500,
    toHaveScreenshot: {
      animations: 'disabled',
      maxDiffPixelRatio: 0.003,
    },
  },
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI
    ? [['line'], ['html', { outputFolder: 'playwright-report', open: 'never' }]]
    : 'line',
  use: {
    baseURL,
    colorScheme: 'light',
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: 'webkit-smoke',
      grep: /@webkit/,
      use: {
        ...devices['Desktop Safari'],
        viewport: { width: 1440, height: 900 },
      },
    },
  ],
  webServer: {
    command: process.env.PLAYWRIGHT_WEB_SERVER_COMMAND
      || `uv run python -m uvicorn web.app:app --host 127.0.0.1 --port ${port}`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: 'ignore',
    stderr: 'pipe',
    env: {
      ...process.env,
      APP_ENV: 'testing',
      SESSION_SECRET: 'playwright-local-only-session-secret',
    },
  },
});
