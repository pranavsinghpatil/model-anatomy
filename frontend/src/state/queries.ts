import { useQuery } from "@tanstack/react-query";
import { api, pending } from "../api/client";

export type ServiceCapability =
  "checking" | "ready" | "stored-only" | "unavailable";

export function useService() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 30_000,
  });
  const service = health.error ? undefined : health.data;
  const available =
    service?.checks.database === true && service.checks.storage === true;
  let capability: ServiceCapability;
  if (health.isPending && !service) capability = "checking";
  else if (!available) capability = "unavailable";
  else if (service?.analysis_available && service.checks.worker)
    capability = "ready";
  else capability = "stored-only";
  return { health, service, available, capability };
}

export function useRecent() {
  return useQuery({
    queryKey: ["analyses"],
    queryFn: ({ signal }) => api.analyses(signal),
    refetchInterval: (q) =>
      q.state.data?.analyses.some((job) => pending(job.status)) ? 3000 : false,
  });
}
