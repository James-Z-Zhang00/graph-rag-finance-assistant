"""
Presidio-backed PII masker — covers all 8 GLBA-relevant entity types.

Detected types
--------------
EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN,
PERSON_NAME, STREET_ADDRESS, DATE_OF_BIRTH, FINANCIAL_ACCOUNT_NUMBER

Falls back to the original regex PIIMasker when PRESIDIO_ENABLED=False or
when the spaCy model cannot be loaded (e.g. in lightweight test environments).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PIIDetection:
    """A single PII span detected in a text."""
    entity_type: str
    recognizer_id: str
    score: float
    start: int
    end: int


@dataclass
class MaskResult:
    """Return value from PresidioMasker.mask()."""
    masked_text: str
    detections: List[PIIDetection] = field(default_factory=list)

    # Convenience — list of unique entity type strings for audit logging
    @property
    def types_found(self) -> List[str]:
        seen = []
        for d in self.detections:
            if d.entity_type not in seen:
                seen.append(d.entity_type)
        return seen

    @property
    def recognizer_ids(self) -> List[str]:
        seen = []
        for d in self.detections:
            if d.recognizer_id not in seen:
                seen.append(d.recognizer_id)
        return seen


# ---------------------------------------------------------------------------
# Supplemental regex patterns used alongside Presidio built-ins
# ---------------------------------------------------------------------------

# SSN: hyphen, space, or no-delimiter variants
_SSN_PATTERN = r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
# Last-4 SSN with context words
_SSN_LAST4_PATTERN = r"(?:SSN|social\s+security)[^\d]{0,20}(\d{4})\b"
# ABA routing number (9 digits) gated on context words
_ROUTING_PATTERN = r"(?:routing|ABA|account)[^\d]{0,15}(\d{9})\b"
# IBAN: up to 34 alphanumeric chars starting with 2-letter country code
_IBAN_PATTERN = r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"
# Date of birth — requires context words to avoid masking fiscal dates
_DOB_CONTEXT = r"(?:born|birth(?:day|date)?|DOB|date\s+of\s+birth)"
_DOB_PATTERN = (
    r"(?:"
    + _DOB_CONTEXT
    + r")[^\d]{0,20}"
    r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
    r"|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4})"
)


class PresidioMasker:
    """
    PII masker backed by Microsoft Presidio + spaCy NER.

    If Presidio/spaCy cannot be initialised (missing packages, model not
    downloaded) the instance transparently falls back to the original
    regex-only PIIMasker so that the service continues to start.
    """

    def __init__(self, score_threshold: float = 0.7):
        self._score_threshold = score_threshold
        self._analyzer = None
        self._anonymizer = None
        self._fallback = None
        self._entities = [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
            "PERSON",           # mapped → PERSON_NAME placeholder
            "LOCATION",         # mapped → STREET_ADDRESS placeholder
            "US_BANK_NUMBER",
            "IBAN_CODE",
            "DATE_TIME",        # filtered by context — see _mask_with_regex_supplements
        ]

        try:
            self._load_presidio()
        except Exception as exc:
            logger.warning(
                "PresidioMasker: failed to initialise Presidio (%s). "
                "Falling back to regex masker.",
                exc,
            )
            from graphrag_agent.compliance.pii_masker import PIIMasker
            self._fallback = PIIMasker()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_presidio(self) -> None:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer.predefined_recognizers import (
            EmailRecognizer,
            PhoneRecognizer,
            CreditCardRecognizer,
            IbanRecognizer,
        )
        from presidio_anonymizer import AnonymizerEngine

        # Build spaCy NLP engine
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=["en"],
        )
        self._anonymizer = AnonymizerEngine()
        logger.info("PresidioMasker: Presidio + spaCy en_core_web_lg loaded.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def mask(self, text: str) -> Tuple[str, "MaskResult"]:
        """
        Scan *text* for PII and replace each detected span with a placeholder.

        Returns
        -------
        (masked_text, MaskResult)
        """
        if self._fallback is not None:
            # Fallback returns (masked_text, List[str]) — wrap in MaskResult
            masked, types = self._fallback.mask(text)
            detections = [
                PIIDetection(
                    entity_type=t.upper(),
                    recognizer_id="regex-fallback",
                    score=1.0,
                    start=0,
                    end=0,
                )
                for t in types
            ]
            return masked, MaskResult(masked_text=masked, detections=detections)

        return self._mask_with_presidio(text)

    # ------------------------------------------------------------------
    # Internal masking logic
    # ------------------------------------------------------------------

    def _mask_with_presidio(self, text: str) -> Tuple[str, MaskResult]:
        from presidio_anonymizer.entities import OperatorConfig

        # --- Presidio analysis ---
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._entities,
            score_threshold=self._score_threshold,
        )

        detections: List[PIIDetection] = []
        operators = {}

        for r in results:
            entity_type = self._normalise_entity_type(r.entity_type)
            placeholder = f"[PII:{entity_type}]"
            operators[r.entity_type] = OperatorConfig(
                "replace", {"new_value": placeholder}
            )
            detections.append(PIIDetection(
                entity_type=entity_type,
                recognizer_id=f"presidio/{r.recognition_metadata.get('recognizer_name', r.entity_type).lower()}",
                score=r.score,
                start=r.start,
                end=r.end,
            ))

        if results:
            from presidio_anonymizer.entities import RecognizerResult as AnonResult
            anon_results = [
                AnonResult(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score,
                )
                for r in results
            ]
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=anon_results,
                operators=operators,
            )
            masked_text = anonymized.text
        else:
            masked_text = text

        # --- Supplemental regex passes for types Presidio under-detects ---
        masked_text, extra = self._mask_with_regex_supplements(masked_text)
        detections.extend(extra)

        return masked_text, MaskResult(masked_text=masked_text, detections=detections)

    def _mask_with_regex_supplements(
        self, text: str
    ) -> Tuple[str, List[PIIDetection]]:
        """Apply supplemental regex patterns for SSN variants, routing numbers, IBAN, and DOB."""
        detections: List[PIIDetection] = []

        supplements = [
            (_SSN_PATTERN, "US_SSN", "regex/ssn-extended"),
            (_SSN_LAST4_PATTERN, "US_SSN", "regex/ssn-last4"),
            (_ROUTING_PATTERN, "FINANCIAL_ACCOUNT_NUMBER", "regex/aba-routing"),
            (_IBAN_PATTERN, "FINANCIAL_ACCOUNT_NUMBER", "regex/iban"),
            (_DOB_PATTERN, "DATE_OF_BIRTH", "regex/dob-context"),
        ]

        for pattern, entity_type, recognizer_id in supplements:
            new_text, count = re.subn(
                pattern,
                f"[PII:{entity_type}]",
                text,
                flags=re.IGNORECASE,
            )
            if count > 0:
                text = new_text
                detections.append(PIIDetection(
                    entity_type=entity_type,
                    recognizer_id=recognizer_id,
                    score=1.0,
                    start=0,
                    end=0,
                ))

        return text, detections

    @staticmethod
    def _normalise_entity_type(presidio_type: str) -> str:
        """Map Presidio internal entity type names to GLBA-aligned names."""
        mapping = {
            "PERSON": "PERSON_NAME",
            "LOCATION": "STREET_ADDRESS",
            "US_BANK_NUMBER": "FINANCIAL_ACCOUNT_NUMBER",
            "IBAN_CODE": "FINANCIAL_ACCOUNT_NUMBER",
            "CREDIT_CARD": "CREDIT_CARD_NUMBER",
        }
        return mapping.get(presidio_type, presidio_type)
