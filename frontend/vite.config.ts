import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath } from "node:url";

export default defineConfig(({ mode }) => {
  const frontendRoot = fileURLToPath(new URL(".", import.meta.url));
  const env = loadEnv(mode, frontendRoot, "");
  const rawPort = env.FRONTEND_PORT || "5173";
  const port = Number(rawPort);
  if (
    !/^\d+$/.test(rawPort) ||
    !Number.isInteger(port) ||
    port < 1024 ||
    port > 65535
  )
    throw new Error("FRONTEND_PORT must be an integer from 1024 to 65535.");
  return {
    root: frontendRoot,
    envDir: frontendRoot,
    plugins: [react(), tailwindcss()],
    server: {
      port,
      strictPort: true,
      watch: {
        ignored: [
          "**/test-results*/**",
          "**/public/licenses/**",
          "**/playwright-report/**",
          "**/.logs/**",
          "**/dist/**",
        ],
      },
      proxy: {
        "/health": {
          target: env.API_PROXY_TARGET || "http://127.0.0.1:8001",
          changeOrigin: true,
        },
        "/api": {
          target: env.API_PROXY_TARGET || "http://127.0.0.1:8001",
          changeOrigin: true,
        },
      },
    },
  };
});
