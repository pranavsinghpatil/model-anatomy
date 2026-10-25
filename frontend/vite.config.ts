import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, searchForWorkspaceRoot } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

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
    cacheDir: ".local/vite-cache",
    server: {
      port,
      strictPort: true,
      fs: {
        allow: [
          searchForWorkspaceRoot(fileURLToPath(new URL(".", import.meta.url))),
          ...["inter", "jetbrains-mono"].map((font) =>
            realpathSync(
              new URL(
                `./node_modules/@fontsource-variable/${font}`,
                import.meta.url,
              ),
            ),
          ),
        ],
      },
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
