# UAPpress Research Engine — Playwright UI Tests + Screenshot Baselines
# Copy each section into the matching file path in your GitHub repo.
# (Single downloadable bundle file)
#
# ✅ What this adds:
# - Playwright E2E UI tests that launch Streamlit and click the app like a user
# - Screenshot baseline snapshots (visual regression)
# - GitHub Actions workflow to run on every push/PR (so you catch UI regressions immediately)
#
# Notes:
# - These tests assume your Streamlit app entrypoint is: apps/research/app.py
# - Streamlit will run on port 8501 during tests.
# - Baseline screenshots are generated with: npx playwright test --update-snapshots
#
# -------------------------------------------------------------------
# FILE: package.json
# -------------------------------------------------------------------
{
  "name": "uappress-ui-tests",
  "private": true,
  "version": "1.0.0",
  "description": "Playwright UI tests + screenshot baselines for UAPpress Streamlit apps",
  "scripts": {
    "test:ui": "playwright test",
    "test:ui:update": "playwright test --update-snapshots",
    "pw:install": "playwright install --with-deps chromium"
  },
  "devDependencies": {
    "@playwright/test": "^1.49.1",
    "typescript": "^5.6.3"
  }
}

# -------------------------------------------------------------------
# FILE: playwright.config.ts
# -------------------------------------------------------------------
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 2 : 0,
  fullyParallel: true,

  reporter: process.env.CI
    ? [["github"], ["html", { open: "never" }]]
    : [["list"], ["html", { open: "never" }]],

  use: {
    baseURL: "http://127.0.0.1:8501",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure"
  },

  // Launch Streamlit automatically before tests.
  webServer: {
    command: "python -m streamlit run apps/research/app.py --server.port 8501 --server.headless true --server.address 127.0.0.1",
    url: "http://127.0.0.1:8501",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe"
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});

# -------------------------------------------------------------------
# FILE: tests/research-ui.spec.ts
# -------------------------------------------------------------------
import { test, expect } from "@playwright/test";

/**
 * UAPpress Research Engine — UI rules we enforce:
 * 1) The Primary topic input is visible.
 * 2) The Run Research button exists and is clickable.
 * 3) Clicking Run Research without required inputs shows an error (client-side validation).
 * 4) A successful run (with minimal inputs) progresses and produces output JSON / download buttons.
 * 5) Visual regression: page layout matches baseline (screenshot snapshots).
 *
 * IMPORTANT:
 * - These tests do NOT require real SerpAPI usage for the "validation" test.
 * - For the "happy path" test, we allow skipping if SERPAPI_API_KEY is not set (prevents wasted credits).
 */

test.describe("UAPpress Research Engine — Smoke UI", () => {
  test("loads and matches baseline (top-of-page)", async ({ page }) => {
    await page.goto("/");
    // Wait for Streamlit to settle
    await page.waitForTimeout(1200);

    // Key UI elements
    await expect(page.getByRole("heading", { name: /UAPpress Research Engine/i })).toBeVisible();
    await expect(page.getByLabel("Primary topic (required)")).toBeVisible();
    await expect(page.getByRole("button", { name: "Run Research" })).toBeVisible();

    // Screenshot baseline — top viewport
    await expect(page).toHaveScreenshot("research-engine-top.png", { fullPage: false });
  });

  test("shows validation error when Primary topic is missing", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(800);

    // Ensure topic is empty
    const topic = page.getByLabel("Primary topic (required)");
    await topic.fill("");

    await page.getByRole("button", { name: "Run Research" }).click();

    // Should show "Primary topic is required."
    await expect(page.getByText("Primary topic is required.", { exact: true })).toBeVisible();

    // Screenshot baseline — validation state
    await expect(page).toHaveScreenshot("research-engine-validation.png", { fullPage: false });
  });

  test("happy path run produces output (skips if SERPAPI_API_KEY missing)", async ({ page }) => {
    test.skip(!process.env.SERPAPI_API_KEY, "SERPAPI_API_KEY not set; skipping to avoid failures/credit burn.");

    await page.goto("/");
    await page.waitForTimeout(800);

    // Fill minimal valid inputs
    await page.getByLabel("Primary topic (required)").fill("Nimitz Tic Tac 2004");
    await page.getByLabel("Time scope (optional)").fill("2004");
    await page.getByLabel("Geo scope (optional)").fill("Pacific");
    await page.getByLabel("Event focus (optional)").fill("DoD statements hearing report");

    // Provide the key into sidebar input (if not already present as env, app may still show empty)
    // Streamlit sidebar fields are still accessible via label
    const serp = page.getByLabel("SerpAPI key (SERPAPI_API_KEY)");
    // If the app already has env value, leave it. Otherwise fill with env.
    const current = await serp.inputValue().catch(() => "");
    if (!current) {
      await serp.fill(process.env.SERPAPI_API_KEY || "");
    }

    await page.getByRole("button", { name: "Run Research" }).click();

    // Expect progress messages
    await expect(page.getByText(/Initializing run/i)).toBeVisible({ timeout: 30_000 });

    // Eventually, we expect the output JSON panel to show metrics/status.
    // Streamlit renders JSON as pre/code blocks; look for known keys.
    await expect(page.getByText(/confidence_overall/i)).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText(/runtime_seconds/i)).toBeVisible({ timeout: 90_000 });

    // Download buttons exist
    await expect(page.getByRole("button", { name: /Download dossier JSON/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Download sources\.csv/i })).toBeVisible();

    // Screenshot baseline — output state (full page)
    await expect(page).toHaveScreenshot("research-engine-after-run.png", { fullPage: true });
  });
});

# -------------------------------------------------------------------
# FILE: .github/workflows/ui-tests.yml
# -------------------------------------------------------------------
name: UI Tests (Playwright)

on:
  push:
    branches: ["main", "master"]
  pull_request:

jobs:
  playwright:
    runs-on: ubuntu-latest

    env:
      # Provide keys via GitHub Actions secrets when you want happy-path to run.
      # If not set, the happy-path test is skipped (validation + baseline tests still run).
      SERPAPI_API_KEY: ${{ secrets.SERPAPI_API_KEY }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          # If you have requirements.txt at repo root, prefer: pip install -r requirements.txt
          # Otherwise install minimal required deps:
          pip install streamlit requests

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"

      - name: Install Node deps
        run: |
          npm ci

      - name: Install Playwright Chromium
        run: |
          npx playwright install --with-deps chromium

      - name: Run Playwright tests
        run: |
          npm run test:ui

      - name: Upload Playwright report (always)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report
          retention-days: 7

# -------------------------------------------------------------------
# FILE: README_UI_TESTS.md  (optional but recommended)
# -------------------------------------------------------------------
# UAPpress — Playwright UI Tests

## One-time setup (Droplet or local)
```bash
# from repo root
npm ci
npx playwright install --with-deps chromium
```

## Run tests
```bash
npm run test:ui
```

## Create / update screenshot baselines
```bash
npm run test:ui:update
```

This writes snapshot images under:
- `tests/research-ui.spec.ts-snapshots/`

Commit those snapshots to GitHub so UI regressions are caught on every push.

## Optional: enable the happy-path test
Set `SERPAPI_API_KEY` in your environment (or as a GitHub secret) so the run test can execute:
```bash
export SERPAPI_API_KEY="your_key"
npm run test:ui
```

## What fails fast
- Missing Primary Topic error behavior
- Missing Run Research button
- Visual layout changes (baseline screenshots)
- Output panel missing expected keys after run (when SERPAPI_API_KEY is present)
