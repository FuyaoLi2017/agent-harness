import unittest

from usage import UsageTotals, format_usage


class UsageTotalsTests(unittest.TestCase):
    def test_haiku_prices_each_response_at_its_full_prompt_size(self):
        totals = UsageTotals()
        usage = {
            "input_tokens": 40_000,
            "cache_creation_input_tokens": 20_000,
            "cache_read_input_tokens": 40_000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 10_000,
                "ephemeral_1h_input_tokens": 10_000,
            },
            "output_tokens": 100,
            "output_tokens_details": {"thinking_tokens": 40},
            "inference_geo": "global",
            "service_tier": "standard",
            "speed": "standard",
        }
        totals.add("claude-haiku-5-5", usage)
        totals.add("claude-haiku-5-5", {**usage, "cache_read_input_tokens": 40_001})

        self.assertEqual(
            totals.summary(),
            {
                "usage": {
                    "response_count": 2,
                    "input_tokens": 80_000,
                    "cache_creation_input_tokens": 40_000,
                    "cache_read_input_tokens": 80_001,
                    "output_tokens": 200,
                    "total_tokens": 200_201,
                },
                "cost": {"estimated_usd": "0.04620005", "pricing_as_of": "2026-11-27"},
            },
        )

    def test_opus_prices_split_cache_writes_from_the_breakdown(self):
        totals = UsageTotals()
        totals.add(
            "claude-opus-5-5",
            {
                "input_tokens": 100,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 10,
                    "ephemeral_1h_input_tokens": 20,
                },
                "cache_read_input_tokens": 40,
                "output_tokens": 15,
            },
        )

        summary = totals.summary()
        self.assertEqual(summary["usage"]["cache_creation_input_tokens"], 30)
        self.assertEqual(summary["cost"]["estimated_usd"], "0.000918")

    def test_incomplete_or_unsupported_pricing_preserves_known_counts(self):
        cases = [
            ("unsupported-model", {"input_tokens": 4}, 4),
            ("claude-opus-5-5", {}, 0),
            ("claude-opus-5-5", {"input_tokens": True}, 0),
            ("claude-opus-5-5", {"input_tokens": 4, "inference_geo": "us"}, 4),
            ("claude-opus-5-5", {"input_tokens": 4, "speed": "fast"}, 4),
            ("claude-opus-5-5", {"input_tokens": 4, "service_tier": "batch"}, 4),
            (
                "claude-opus-5-5",
                {"input_tokens": 4, "server_tool_use": {"web_search_requests": 1}},
                4,
            ),
            ("claude-opus-5-5", {"input_tokens": 4, "cache_read_input_tokens": "2"}, 4),
        ]
        for model, usage, known_input in cases:
            with self.subTest(model=model, usage=usage):
                totals = UsageTotals()
                totals.add(model, {**usage, "output_tokens": 3})
                totals.add("claude-haiku-5-5", {"input_tokens": 10, "output_tokens": 5})

                summary = totals.summary()
                self.assertIsNone(summary["cost"]["estimated_usd"])
                self.assertEqual(summary["usage"]["response_count"], 2)
                self.assertEqual(summary["usage"]["input_tokens"], 10 + known_input)
                self.assertEqual(summary["usage"]["output_tokens"], 8)

    def test_cache_write_count_survives_unavailable_ttl_pricing(self):
        for breakdown in (None, {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 4}):
            with self.subTest(breakdown=breakdown):
                totals = UsageTotals()
                totals.add(
                    "claude-opus-5-5",
                    {
                        "input_tokens": 4,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 15,
                        "cache_creation": breakdown,
                    },
                )

                summary = totals.summary()
                self.assertIsNone(summary["cost"]["estimated_usd"])
                self.assertEqual(summary["usage"]["cache_creation_input_tokens"], 15)

    def test_empty_usage_has_no_estimate(self):
        totals = UsageTotals()

        empty_summary = totals.summary()
        self.assertEqual(empty_summary["usage"]["response_count"], 0)
        self.assertEqual(empty_summary["usage"]["total_tokens"], 0)
        self.assertIsNone(empty_summary["cost"]["estimated_usd"])

        totals.add("claude-haiku-5-5", {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(totals.summary()["cost"]["estimated_usd"], "0")

    def test_format_attributes_replay_estimate_to_the_original_run(self):
        totals = UsageTotals()
        totals.add("claude-haiku-5-5", {"input_tokens": 4, "output_tokens": 3})

        live = format_usage(totals.summary(), replay=False)
        replay = format_usage(totals.summary(), replay=True)

        self.assertIn("1 response", replay)
        self.assertIn("7 total", replay)
        self.assertIn("$0.0000019", replay)
        self.assertIn("2026-11-27", replay)
        self.assertIn("original run", replay)
        self.assertNotIn("original run", live)
        self.assertIn("unavailable", format_usage(UsageTotals().summary(), replay=False))
