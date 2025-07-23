# AIDA/runbooks/cpu_usage.md

# Runbook: Diagnosing High CPU Usage in Kubernetes

## Symptoms

- A Prometheus alert fires for `HighCpuUsage`.
- Pods may become slow or unresponsive.
- Pods might be getting restarted by Kubernetes if they exceed their CPU limits.

## Diagnostic Steps

1.  **Identify the affected pod:** Use the `alert` information to find the name of the pod experiencing high CPU.
2.  **Describe the pod:** Use `kubectl describe pod <pod_name>` to check for recent events, restarts, or configuration issues. Pay close attention to the `Events` section at the bottom for anomalies.
3.  **Check resource utilization:** Use `kubectl top pod <pod_name>` to see the current, real-time CPU and memory usage. Compare this to the resource `requests` and `limits` defined in the pod's specification.
4.  **Inspect logs:** Use a Loki or OpenSearch query to look for error messages, stack traces, or unusually high volumes of log output that might indicate an infinite loop or processing-intensive task. A common cause is a memory leak leading to excessive garbage collection cycles, which consumes CPU. Another possibility is a timeout when connecting to a downstream service, causing repeated, CPU-intensive retry loops.
