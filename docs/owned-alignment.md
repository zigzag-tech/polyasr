# Owned forced alignment (source qualification)

The CUDA service adds an optional owned job API while preserving existing `/v1/align`, `/v1/align/manifest`, transcription, streaming, profiles and model defaults. MLX does not advertise this protocol.

- `GET /v1/align/capabilities`: protocol version 1, instance ID, process-memory receipt retention, upload transport, physical cancellation scope `alignment`, unbounded latency.
- `POST /v1/align/jobs`: multipart `file` plus `submission` JSON `{job_id,owner_id,control_token,request}`. Client supplies UUID and private token before submission. Request is `{audio_sha256,language?,max_chunk_seconds?,model?}`. Upload SHA256 must match; request identity is sorted compact JSON SHA256. Served model remains authoritative as with legacy endpoints.
- `GET /v1/align/jobs/{id}`: ownership and physical-settlement receipt.
- `POST /v1/align/jobs/{id}/cancel`: `{owner_id}`; supports cancel-before-submit tombstones.
- `GET /v1/align/jobs/{id}/result`: exact settled alignment JSON.

Status, cancel and result require `Authorization: Bearer <control_token>`. Receipts never contain that token. Preserve credentials in a private owner store and persist a public prepared effect in the authoritative workflow journal before submit. Identical job/request retries replay; changed content or owner refuses.

A cancel request sets a cooperative signal. It prevents model loading while a queued task waits for the existing model lock and stops before the next chunk. An active model call may finish first. Neither HTTP closure nor a cancellation request proves physical settlement. The service publishes `physical_settled:true` only after the owning executor task has finished and its CUDA synchronization barrier succeeds. Barrier failure remains unsettled. Owned uploads are removed on completion, queued cancellation, duplicate submit and tombstones. Other users' work is never killed.

Control requests should have a 20-second transport bound. Model execution may take longer; the owner must retain its Harmony lease until physical settlement. Forced alignment uses the `align` managed unit, so strict clients must admit/lease that unit rather than the separate streaming `asr` unit. Legacy clients retain their existing policy.

This API depends on `livestack_node.owned_executor_jobs`. Source tests exercise both PolyTTS's compatibility adapter and this ASR adapter over that shared runtime. No service deployment or live GPU qualification has occurred. Process restart loses receipts: an unknown accepted job must remain held for explicit reconciliation rather than be silently resubmitted. The current Unchain client still needs the new owned upload/status protocol and durable ASR effect adapter before live strict audio is qualified.
