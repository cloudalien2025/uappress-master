UAPpress Playwright Fix Bundle (No-Tests-Found Fix)
==================================================

Your GitHub Action failed with:
  Error: No tests found

Cause:
- Your tests/research-ui.spec.ts begins with "#" comment lines (copied from a bundle header).
  That makes the file invalid TypeScript, so Playwright can't discover tests.

Fix:
- Replace the contents of the files below EXACTLY with the contents in this bundle.

----------------------------------------------------------------------
FILE: package.json
PATH: /package.json
----------------------------------------------------------------------

{
  "name": "uappress-master",
  "private": true,
  "version": "1.0.0",
  "description": "Playwright UI tests for UAPpress Streamlit apps",
  "scripts": {
    "test:ui": "playwright test",
    "test:ui:headed": "playwright test --headed",
    "test:ui:debug": "playwright test --debug",
    "test:ui:update": "playwright test --update-snapshots"
  },
  "devDependencies": {
    "@playwright/test": "^1.41.2"
  }
}

----------------------------------------------------------------------
FILE: playwright.config.ts
PATH: /playwright.config.ts
----------------------------------------------------------------------

import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.BASE_URL || "http://127.0.0.1:8501";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: [["html", { open: "never" }], ["list"]],
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure"
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});

----------------------------------------------------------------------
FILE: research-ui.spec.ts
PATH: /tests/research-ui.spec.ts
----------------------------------------------------------------------

import { test, expect } from "@playwright/test";

/**
 * UAPpress Research Engine — UI Smoke Tests
 *
 * Assumes BASE_URL points to a running Streamlit server.
 * GitHub Actions currently points BASE_URL at your Droplet URL.
 */

test.describe("UAPpress Research Engine UI", () => {
  test("loads and shows title", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("UAPpress Research Engine — SerpAPI (Two-Pass)")).toBeVisible();
  });

  test("Run Research button is visible", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Run Research" })).toBeVisible();
  });

  test("Primary topic input exists", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("textbox", { name: /Primary topic/i })).toBeVisible();
  });
});

----------------------------------------------------------------------
FILE: ui-tests.yml
PATH: /.github/workflows/ui-tests.yml
----------------------------------------------------------------------

name: UI Tests (Playwright)

on:
  push:
    branches: [ "main" ]
  pull_request:

jobs:
  playwright:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright Browsers
        run: npx playwright install --with-deps chromium

      # Runs tests against your deployed Droplet.
      - name: Run UI tests
        env:
          BASE_URL: "http://104.236.44.185:8501"
        run: npm run test:ui

      - name: Upload Playwright report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report
          if-no-files-found: ignore

==================================================
Baby steps checklist
==================================================

1) Open /tests/research-ui.spec.ts → Edit (pencil) → replace everything → Commit.
2) Open /.github/workflows/ui-tests.yml → replace everything → Commit.
3) Open /playwright.config.ts → replace everything → Commit.
4) Open /package.json → replace everything → Commit.

Then go to Actions and the workflow should find tests and run them.

If the next failure is “cannot connect”, it means GitHub Actions cannot reach your droplet URL.
Then we switch to running Streamlit inside CI (I’ll give you that workflow next).
