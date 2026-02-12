import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: {
    baseURL: 'http://104.236.44.185:8501',
    headless: true,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  timeout: 120000,
});
