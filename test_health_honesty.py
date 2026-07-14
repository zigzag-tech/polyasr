"""Health must not report green while the streaming path is dead.

Regression tests for the 2026-07-14 outage: the HF cache was wiped, Harmony
evicted the idle model, the reload failed, and every session returned empty text
for 35 minutes while /health served {"status": "ok", "healthy": true}.

See docs/health-honesty.md.
"""
import unittest

from server import compute_health

THRESH = 3
STALE = 180.0


def health(**kw):
    args = dict(
        weights_present=True,
        probe_enabled=True,
        consecutive_failures=0,
        fail_threshold=THRESH,
        last_ok_age_sec=1.0,
        stale_after_sec=STALE,
    )
    args.update(kw)
    return compute_health(**args)


class ComputeHealth(unittest.TestCase):
    def test_healthy_server_is_ok(self):
        self.assertEqual(health(), ("ok", []))

    def test_wiped_weights_are_degraded(self):
        # THE INCIDENT: weights deleted from the HF cache. The model transcribes
        # to empty (which looks exactly like silence), so nothing else notices.
        status, reasons = health(weights_present=False)
        self.assertEqual(status, "degraded")
        self.assertIn("weights_missing", reasons)

    def test_incident_state_exactly(self):
        # The literal /health payload observed during the outage: zero probe
        # failures (the probe had stopped running), a stale successful result,
        # and weights gone. This MUST be degraded — it used to report "ok".
        status, reasons = health(
            weights_present=False,
            consecutive_failures=0,   # probe never ran => no failures recorded
            last_ok_age_sec=2400.0,   # last success was 40 minutes ago
        )
        self.assertEqual(status, "degraded")
        self.assertIn("weights_missing", reasons)
        self.assertIn("streaming_probe_stale", reasons)

    def test_stale_probe_is_not_health(self):
        # Absence of evidence is not evidence of health.
        status, reasons = health(last_ok_age_sec=STALE + 1)
        self.assertEqual(status, "degraded")
        self.assertIn("streaming_probe_stale", reasons)

    def test_probe_that_never_succeeded_is_not_health(self):
        status, reasons = health(last_ok_age_sec=None)
        self.assertEqual(status, "degraded")
        self.assertIn("streaming_probe_stale", reasons)

    def test_failing_probe_is_degraded(self):
        # The silent partial-window rot: probe returns empty for canonical speech.
        status, reasons = health(consecutive_failures=THRESH)
        self.assertEqual(status, "degraded")
        self.assertIn("streaming_probe_failing", reasons)

    def test_failures_below_threshold_stay_ok(self):
        # Hysteresis: one blank probe is not an outage (noise, onset, cold start).
        self.assertEqual(health(consecutive_failures=THRESH - 1), ("ok", []))

    def test_evicted_model_with_good_weights_is_ok(self):
        # A legitimately evicted unit is NOT unhealthy — we don't resurrect it
        # just to probe. The weights check is the health signal while evicted,
        # so the probe is not expected to be fresh.
        self.assertEqual(
            health(probe_enabled=False, last_ok_age_sec=9999.0, consecutive_failures=99),
            ("ok", []),
        )

    def test_evicted_model_with_missing_weights_is_degraded(self):
        # ...but an evicted model that can no longer load is exactly the outage.
        status, reasons = health(
            probe_enabled=False, weights_present=False, last_ok_age_sec=9999.0
        )
        self.assertEqual(status, "degraded")
        self.assertEqual(reasons, ["weights_missing"])

    def test_staleness_disabled_when_probe_unscheduled(self):
        self.assertEqual(health(stale_after_sec=None, last_ok_age_sec=None), ("ok", []))


if __name__ == "__main__":
    unittest.main()
