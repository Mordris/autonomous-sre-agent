# AIDA/runbooks/crash_loop.md

# Runbook: Diagnosing CrashLoopBackOff in Kubernetes

## Symptoms

- A Prometheus alert fires for `KubePodCrashLooping`.
- `kubectl get pods` shows a pod with the status `CrashLoopBackOff`.

## Diagnostic Steps

1.  **Check pod logs:** This is the most critical step. The application inside the container is exiting with an error. You need to see that error. Use `kubectl logs <pod_name>` to see the logs from the _current_, crashing container.
2.  **Check logs of the previous container:** A `CrashLoopBackOff` means the container has crashed multiple times. To see why it crashed on the previous attempt, use `kubectl logs <pod_name> --previous`. This is often where the real startup error is found.
3.  **Describe the pod:** Use `kubectl describe pod <pod_name>`. The `Events` section will show the lifecycle of the pod, including start-up probes failing, readiness probes failing, and back-off delay warnings. It might also show OOMKilled (Out of Memory) events if the container exceeded its memory limit.
4.  **Check configuration:** Verify that `ConfigMaps` and `Secrets` the pod depends on are mounted correctly and contain the right values. An error in configuration, like a missing environment variable or a wrong database URL, is a common reason for applications to fail on startup.
