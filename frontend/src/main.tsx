import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { retryQuery } from "./api/client";
import { App } from "./App";
import { LazyMotion } from "motion/react";
import "@xyflow/react/dist/style.css";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles.css";

const client = new QueryClient({
  defaultOptions: {
    queries: {
      retry: retryQuery,
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
    mutations: { retry: false },
  },
});
const loadMotionFeatures = () =>
  import("./components/vendor/motion-features").then(
    (module) => module.default,
  );
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <LazyMotion features={loadMotionFeatures} strict>
        <App />
      </LazyMotion>
    </QueryClientProvider>
  </StrictMode>,
);
