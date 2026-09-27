import { defineConfig } from "vitest/config";

// The bundle lands inside the Python package so `neurogarden serve` needs no Node at run time.
export default defineConfig({
  base: "./",
  build: {
    outDir: "../src/neurogarden/server/static",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
  },
  server: {
    proxy: { "/ws": { target: "ws://127.0.0.1:8765", ws: true, rewrite: () => "/" } },
  },
  test: {
    environment: "node",
  },
});
