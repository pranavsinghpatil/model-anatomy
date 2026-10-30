import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "./ui";
import "./demo-controls.css";

type DemoState = {
  state: "stopped" | "starting" | "running" | "stopping";
  ready: boolean;
  queued: number;
  running: number;
  start_allowed: boolean;
  stop_allowed: boolean;
};

async function control(action?: "start" | "stop"): Promise<DemoState> {
  const response = await fetch("/api/demo", {
    method: action ? "POST" : "GET",
    credentials: "same-origin",
    cache: "no-store",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    ...(action ? { body: JSON.stringify({ action }) } : {}),
    signal: AbortSignal.timeout(action === "start" ? 250_000 : 30_000),
  });
  const value = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(value?.error?.message || "Demo controls are unavailable.");
  if (
    !value ||
    !["stopped", "starting", "running", "stopping"].includes(value.state) ||
    typeof value.ready !== "boolean" ||
    typeof value.start_allowed !== "boolean" ||
    typeof value.stop_allowed !== "boolean" ||
    !Number.isInteger(value.queued) ||
    !Number.isInteger(value.running)
  )
    throw new Error("The demo controls returned an invalid response.");
  return value as DemoState;
}

function EnabledDemoControls() {
  const client = useQueryClient();
  const panel = useRef<HTMLElement>(null);
  const lastReady = useRef<boolean | undefined>(undefined);
  const [message, setMessage] = useState("");
  const status = useQuery({
    queryKey: ["demo-control"],
    queryFn: () => control(),
    retry: false,
    refetchInterval: (query) =>
      query.state.data?.state === "stopped" ? 30_000 : 5_000,
    refetchOnWindowFocus: true,
  });
  const mutation = useMutation({
    mutationFn: control,
    onMutate: async () => {
      setMessage("");
      await client.cancelQueries({ queryKey: ["demo-control"] });
    },
    onSuccess: (data) => {
      client.setQueryData(["demo-control"], data);
      void client.invalidateQueries({ queryKey: ["demo-control"] });
    },
    onError: (error) => {
      setMessage(error.message);
      void client.invalidateQueries({ queryKey: ["demo-control"] });
    },
  });
  const state = status.data;
  useEffect(() => {
    const element = panel.current;
    if (!element) return;
    const update = () =>
      document.documentElement.style.setProperty(
        "--demo-controls-height",
        `${element.offsetHeight}px`,
      );
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => {
      observer.disconnect();
      document.documentElement.style.removeProperty("--demo-controls-height");
    };
  }, []);
  useEffect(() => {
    if (state && lastReady.current !== state.ready) {
      lastReady.current = state.ready;
      void client.invalidateQueries({
        predicate: (query) => query.queryKey[0] !== "demo-control",
      });
    }
  }, [client, state]);
  const labels = {
    stopped: "Demo is stopped. Start it when you are ready.",
    starting: "Starting demo. Please wait for the analysis service.",
    running: "Demo is running.",
    stopping: "Stopping demo and saving its data.",
  };
  const busy = Boolean(state && state.queued + state.running > 0);
  return (
    <section
      ref={panel}
      className="demo-controls"
      aria-label="Shared demo controls"
    >
      <div className="demo-controls-copy">
        <strong>Interactive demo</strong>
        <p role="status" aria-live="polite">
          {status.isError
            ? "Cannot check the demo. Retry to see its current status."
            : state
              ? labels[state.state]
              : "Checking demo status..."}
        </p>
        <small>
          {busy
            ? "An analysis is queued or running. Stop is unavailable."
            : "Shared by all visitors. Stop affects everyone. Sessions last up to 45 minutes."}
        </small>
        {state && !state.start_allowed && state.state === "stopped" && (
          <p>The demo start period has ended.</p>
        )}
        {message && <p role="alert">{message}</p>}
      </div>
      <div className="demo-controls-actions">
        <Button
          variant="primary"
          disabled={
            !state?.start_allowed ||
            state.ready ||
            mutation.isPending ||
            status.isError
          }
          onClick={() => mutation.mutate("start")}
        >
          {mutation.isPending && mutation.variables === "start"
            ? "Starting..."
            : "Start Demo"}
        </Button>
        <Button
          disabled={
            !state?.stop_allowed || mutation.isPending || status.isError
          }
          onClick={() => mutation.mutate("stop")}
        >
          {mutation.isPending && mutation.variables === "stop"
            ? "Stopping..."
            : "Stop Demo"}
        </Button>
        {status.isError && (
          <Button
            disabled={status.isFetching}
            onClick={() => void status.refetch()}
          >
            Retry status
          </Button>
        )}
      </div>
    </section>
  );
}

export function DemoControls() {
  return import.meta.env.VITE_DEMO_CONTROLS === "true" ? (
    <EnabledDemoControls />
  ) : null;
}
