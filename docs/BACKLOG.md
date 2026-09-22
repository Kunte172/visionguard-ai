# Backlog

## v0.2 — benchmark representation model

- Add an opt-in MVTec AD adapter without redistributing the dataset.
- Implement a PatchCore-style pretrained feature extractor and coreset memory.
- Report per-category image/pixel AUROC and PRO with repeatable environment metadata.
- Add GPU/CPU latency and memory benchmarks against the v0.1 baseline.

## v0.3 — pilot hardening

- Version datasets and model artifacts in object storage plus a registry.
- Add camera/product preprocessing profiles and threshold approval history.
- Export traces, metrics, structured logs, and drift statistics.
- Add authentication, rate limiting, malware-aware uploads, and retention policy.
- Support batch jobs, async processing, review queues, and operator feedback.
- Add shadow deployment, canary rollout, rollback, and failure-mode runbooks.

## Interview discussion prompts

- Why normal-only training is useful when defects are rare and changing.
- Why AUROC alone is insufficient for a production quality gate.
- How camera drift changes score distributions and threshold validity.
- Which API boundaries stay stable when replacing the baseline with PatchCore.
