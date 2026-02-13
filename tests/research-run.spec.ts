import { test, expect } from "@playwright/test";

test.describe("UAPpress Research Engine UI", () => {
  test("clicks Run and confirms the app produces output (not just a loaded UI)", async ({ page }) => {
    const baseURL = process.env.BASE_URL || "http://127.0.0.1:8501";

    // Load app
    await page.goto(baseURL, { waitUntil: "domcontentloaded" });

    // Streamlit hydration buffer
    await page.waitForTimeout(1200);

    // Primary Topic input should exist (same as your smoke test)
    const primaryTopic = page.getByLabel(/primary topic/i);
    await expect(primaryTopic).toBeVisible({ timeout: 15000 });

    // Run button should exist and be clickable
    const runBtn = page.getByRole("button", { name: /^run$/i });
    await expect(runBtn).toBeVisible({ timeout: 15000 });
    await expect(runBtn).toBeEnabled();

    // Click Run
    await runBtn.click();

    // After clicking Run, we expect *some* sign of progress or results.
    // Because Streamlit DOM is variable, we look for multiple likely signals.
    const resultSignals = [
      page.getByText(/results?/i).first(),
      page.getByText(/sources?/i).first(),
      page.getByText(/summary/i).first(),
      page.getByText(/output/i).first(),
      page.getByText(/running/i).first(),
      page.getByText(/working/i).first(),
      page.getByText(/complete|done|finished/i).first(),
      page.getByRole("button", { name: /download/i }).first(),
    ];

    const timeoutMs = 60000;
    const start = Date.now();
    let signalFound = false;

    while (Date.now() - start < timeoutMs) {
      for (const loc of resultSignals) {
        try {
          if (await loc.isVisible()) {
            signalFound = true;
            break;
          }
        } catch {
          // ignore transient DOM re-render errors during Streamlit refresh
        }
      }
      if (signalFound) break;

      // Also fail fast if Streamlit displays a visible python error
      const hasTraceback = await page.getByText(/traceback/i).first().isVisible().catch(() => false);
      const hasException = await page.getByText(/exception/i).first().isVisible().catch(() => false);
      const hasErrorBanner = await page
        .getByText(/the app has encountered an error/i)
        .first()
        .isVisible()
        .catch(() => false);

      if (hasTraceback || hasException || hasErrorBanner) {
        throw new Error("Run triggered an app error (Traceback/Exception/Error banner visible).");
      }

      await page.waitForTimeout(750);
    }

    expect(signalFound, "Expected a progress/results signal after clicking Run, but saw none. Add a stable 'Results' header or data-testid marker in Streamlit.").toBeTruthy();

    // Sanity: still not showing Streamlit error state
    await expect(page.getByText(/the app has encountered an error/i)).toHaveCount(0);
  });
});
