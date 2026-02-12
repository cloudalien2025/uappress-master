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
