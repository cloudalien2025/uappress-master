// tests/research-run.spec.ts — UAPpress Research Engine (Deterministic, viewport-safe)
import { test, expect } from "@playwright/test";

test.describe("UAPpress Research Engine (Smoke)", () => {
  test("can run deterministic smoke research and render stable outputs", async ({ page, baseURL }) => {
    const url = baseURL || "http://127.0.0.1:8501";
    await page.goto(url, { waitUntil: "domcontentloaded" });

    await expect(page.getByText("TEST_HOOK:APP_LOADED", { exact: false })).toBeVisible();
    await expect(page.getByText("UAPpress Research Engine", { exact: false })).toBeVisible();

    await page.getByLabel("Primary Topic", { exact: true }).fill("Phoenix Lights");
    await page.getByRole("button", { name: "Run Research", exact: true }).click();

    await expect(page.getByText("TEST_HOOK:RUN_DONE", { exact: false })).toBeVisible();
    await expect(page.getByText("Research Complete", { exact: false })).toBeVisible();

    await page.getByRole("button", { name: "Build Fact Pack", exact: true }).click();
    await page.getByRole("button", { name: "Build Beat Sheet", exact: true }).click();
    await page.getByRole("button", { name: "Generate Script", exact: true }).click();

    await expect(page.getByText("Script Quality", { exact: true })).toBeVisible();
    await expect(page.getByText("PASS", { exact: false })).toBeVisible();

    const scriptText = await page.locator('textarea[aria-label="Script (validated)"]').inputValue();
    for (const banned of ["phase", "cycle", "marker", "in this section", "this scene"]) {
      expect(scriptText.toLowerCase()).not.toContain(banned);
    }

    const repeatedPhraseCount = (scriptText.match(/On March 13, 1997/g) || []).length;
    expect(repeatedPhraseCount).toBeLessThanOrEqual(2);

    const sceneCardCount = await page.locator('text=/Voiceover — Scene \\d+/').count();
    expect(sceneCardCount).toBeGreaterThanOrEqual(12);
    expect(sceneCardCount).toBeLessThanOrEqual(20);

    const mp3Button = page.getByRole("button", { name: "Generate MP3s (per scene)", exact: true });
    await expect(mp3Button).toBeEnabled();
  });
});
