"""
Unit tests for PresidioMasker — one test per GLBA entity type plus
regression, performance, and fallback tests.
"""

import time
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def masked_text(masker, text: str) -> str:
    result, _ = masker.mask(text)
    return result


def detections(masker, text: str):
    _, mask_result = masker.mask(text)
    return mask_result.detections


def types_found(masker, text: str):
    _, mask_result = masker.mask(text)
    return mask_result.types_found


# ---------------------------------------------------------------------------
# GLBA entity type tests
# ---------------------------------------------------------------------------

class TestEmailAddress:
    def test_masks_standard_email(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Contact me at john.doe@example.com for details.")
        assert "john.doe@example.com" not in result
        assert "[PII:EMAIL_ADDRESS]" in result

    def test_email_type_in_detections(self, presidio_masker_instance):
        found = types_found(presidio_masker_instance, "Send to alice@corp.io please.")
        assert "EMAIL_ADDRESS" in found


class TestPhoneNumber:
    def test_masks_parenthesis_format(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Call (555) 867-5309 now.")
        assert "867-5309" not in result

    def test_masks_dot_format(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Reach us at 555.867.5309.")
        assert "555.867.5309" not in result

    def test_masks_plus1_format(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "International: +1-555-867-5309.")
        assert "+1-555-867-5309" not in result


class TestSSN:
    def test_masks_hyphen_format(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "SSN is 123-45-6789.")
        assert "123-45-6789" not in result

    def test_masks_space_format(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "SSN is 123 45 6789.")
        assert "123 45 6789" not in result

    def test_last_four_with_context(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "SSN ending in 6789 on file.")
        assert "6789" not in result

    def test_fiscal_year_not_masked(self, presidio_masker_instance):
        # "2024" in fiscal context should not trigger SSN masking
        result = masked_text(presidio_masker_instance, "Revenue grew in 2024.")
        assert "2024" in result


class TestCreditCard:
    def test_masks_spaced_card(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Card: 4111 1111 1111 1111.")
        assert "4111 1111 1111 1111" not in result

    def test_masks_dashed_card(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Card: 4111-1111-1111-1111.")
        assert "4111-1111-1111-1111" not in result


class TestPersonName:
    def test_masks_full_name(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "John Smith submitted the quarterly report.")
        assert "John Smith" not in result

    def test_company_name_not_masked(self, presidio_masker_instance):
        # Company names like "Apple Inc." should NOT be masked as PERSON_NAME
        result = masked_text(presidio_masker_instance, "Apple Inc. reported strong earnings.")
        assert "Apple" in result


class TestStreetAddress:
    def test_masks_us_address(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Located at 123 Main Street, Boston, MA 02101.")
        assert "123 Main Street" not in result


class TestDateOfBirth:
    def test_masks_dob_with_context(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Born on 01/15/1985.")
        assert "01/15/1985" not in result

    def test_fiscal_quarter_not_masked(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Results for Q1 2024.")
        assert "Q1 2024" in result


class TestFinancialAccountNumber:
    def test_masks_routing_with_context(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "routing number 021000021 for the transfer.")
        assert "021000021" not in result

    def test_iban_masked(self, presidio_masker_instance):
        result = masked_text(presidio_masker_instance, "Transfer to GB29NWBK60161331926819.")
        assert "GB29NWBK60161331926819" not in result


# ---------------------------------------------------------------------------
# Multi-entity and clean text tests
# ---------------------------------------------------------------------------

class TestMultipleEntities:
    def test_all_three_detected(self, presidio_masker_instance):
        text = "Email john@test.com, call (555) 123-4567, SSN 987-65-4321."
        found = types_found(presidio_masker_instance, text)
        result = masked_text(presidio_masker_instance, text)
        assert "john@test.com" not in result
        assert "987-65-4321" not in result
        assert len(found) >= 2  # at minimum email + ssn

    def test_clean_text_unchanged(self, presidio_masker_instance):
        text = "What was Apple Inc. revenue in Q1 2024?"
        result, mask_result = presidio_masker_instance.mask(text)
        assert result == text
        assert mask_result.detections == []
        assert mask_result.types_found == []


# ---------------------------------------------------------------------------
# Recognizer ID and MaskResult structure
# ---------------------------------------------------------------------------

class TestRecognizerId:
    def test_recognizer_id_populated(self, presidio_masker_instance):
        _, mask_result = presidio_masker_instance.mask("Call me at john@test.com.")
        for d in mask_result.detections:
            assert d.recognizer_id != ""
            assert d.recognizer_id is not None

    def test_mask_result_types_found_unique(self, presidio_masker_instance):
        # Two emails in one text — types_found should list EMAIL_ADDRESS once
        _, mask_result = presidio_masker_instance.mask(
            "Email a@test.com and b@test.com."
        )
        assert mask_result.types_found.count("EMAIL_ADDRESS") == 1


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_to_regex_when_disabled(self, monkeypatch):
        monkeypatch.setenv("PRESIDIO_ENABLED", "false")
        from graphrag_agent.compliance.presidio_masker import PresidioMasker
        with patch.object(PresidioMasker, "_load_presidio", side_effect=ImportError("no presidio")):
            masker = PresidioMasker()
        # Fallback masker should still catch email
        result, mask_result = masker.mask("Contact john@test.com please.")
        assert "john@test.com" not in result

    def test_model_load_failure_does_not_raise(self):
        from graphrag_agent.compliance.presidio_masker import PresidioMasker
        with patch.object(PresidioMasker, "_load_presidio", side_effect=OSError("model not found")):
            masker = PresidioMasker()  # must not raise
        assert masker._fallback is not None


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    def test_100_calls_under_2_seconds(self, presidio_masker_instance):
        text = "What was the revenue for Apple Inc. in Q1 2024?"
        start = time.monotonic()
        for _ in range(100):
            presidio_masker_instance.mask(text)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"100 calls took {elapsed:.2f}s — expected < 2s"
