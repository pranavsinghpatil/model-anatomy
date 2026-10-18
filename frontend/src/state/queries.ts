import { useQuery } from "@tanstack/react-query";
import { api, pending } from "../api/client";
export function useService() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => api.health(signal),
    refetchInterval: 30_000,
  });
  const service = health.error ? undefined : health.data;
  return {
    health,
    service,
    available:
      service?.status === "ok" &&
      service.checks.database &&
      service.checks.storage &&
      service.checks.worker &&
      service.analysis_available,
  };
}
export function useRecent() {
  return useQuery({
    queryKey: ["analyses"],
    queryFn: ({ signal }) => api.analyses(signal),
    refetchInterval: (q) =>
      q.state.data?.analyses.some((job) => pending(job.status)) ? 3000 : false,
  });
}
