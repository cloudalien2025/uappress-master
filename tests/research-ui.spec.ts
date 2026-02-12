import { test, expect } from "@playwright/test";

test.describe("UAPpress Research Engine UI", () => {
  test("loads the Research Engine and shows primary topic + run button", async ({ page, baseURL }) => {
    // If you run Streamlit multi-app routing, adjust this.
    // For your repo, you are running apps/research/app.py directly in CI
    // so it will be hosted at "/".
    await page.goto(baseURL || "http://127.0.0.1:8501", { waitUntil: "domcontentloaded" });

    // Streamlit can take a moment to hydrate UI
    await page.waitForTimeout(1200);

    // Title / header text (keep these flexible)
    await expect(page.getByText("UAPpress Research Engine", { exact: false })).toBeVisible();

    // Primary topic input exists (label text can vary by Streamlit theme)
    await expect(page.getByText("Primary topic", { exact: false })).toBeVisible();

    // Run button exists
    await expect(page.getByRole("button", { name: /Run Research/i })).toBeVisible();
  });
});
