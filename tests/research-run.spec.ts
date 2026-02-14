// tests/research-run.spec.ts — UAPpress Research Engine (Deterministic)
import { test, expect } from "@playwright/test";

test.describe("UAPpress Research Engine (Smoke)", () => {
  test("can run deterministic smoke research and render stable outputs", async ({ page, baseURL }) => {
    const url = baseURL || "http://127.0.0.1:8501";
    await page.goto(url, { waitUntil: "domcontentloaded" });

    // Streamlit hydration can lag; wait for a stable marker from the app
    await expect(page.getByText("TEST_HOOK:APP_LOADED", { exact: false })).toBeVisible();

    // Title
    await expect(page.getByText("UAPpress Research Engine", { exact: false })).toBeVisible();

    // Fill topic
    await page.getByLabel("Primary Topic", { exact: true }).fill("ODNI UAP Report 2023");

    // Click run (form submit)
    await page.getByRole("button", { name: "Run Research", exact: true }).click();

    // Wait for completion marker
    await expect(page.getByText("TEST_HOOK:RUN_DONE", { exact: false })).toBeVisible();

    // Deterministic assertions (smoke-mode fixture)
    await expect(page.getByText("Research Complete", { exact: false })).toBeVisible();
    await expect(page.getByText("Mock Source A", { exact: false })).toBeVisible();
    await expect(page.getByText("Mock Source B", { exact: false })).toBeVisible();
    await expect(page.getByText("Mock Source C", { exact: false })).toBeVisible();
  });
});
