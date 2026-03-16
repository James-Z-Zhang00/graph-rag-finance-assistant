import re
from typing import List, Tuple

from config.settings import CHUNK_SIZE, OVERLAP, MAX_TEXT_LENGTH

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

class TextChunker:
    """Text chunker that splits long text into overlapping chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, max_text_length: int = MAX_TEXT_LENGTH):
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_text_length = max_text_length
        self.encoding = tiktoken.get_encoding("cl100k_base") if tiktoken else None

    def process_files(self, file_contents: List[Tuple[str, str]]) -> List[Tuple[str, str, List[List[str]]]]:
        results = []
        for filename, content in file_contents:
            chunks = self.chunk_text(content)
            results.append((filename, content, chunks))
        return results

    def _preprocess_large_text(self, text: str) -> List[str]:
        if len(text) <= self.max_text_length:
            return [text]

        target_segment_size = min(self.max_text_length, max(10000, self.max_text_length // 2))
        paragraphs = text.split('\n\n')

        if len(paragraphs) < 5:
            paragraphs = text.split('\n')

        processed_segments = []
        current_segment = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > target_segment_size:
                if current_segment:
                    processed_segments.append(current_segment)
                    current_segment = ""
                split_paras = self._split_long_paragraph(para, target_segment_size)
                processed_segments.extend(split_paras)
            else:
                if len(current_segment) + len(para) + 2 > target_segment_size:
                    if current_segment:
                        processed_segments.append(current_segment)
                    current_segment = para
                else:
                    if current_segment:
                        current_segment += "\n\n" + para
                    else:
                        current_segment = para

        if current_segment:
            processed_segments.append(current_segment)

        return processed_segments

    def _split_long_paragraph(self, text: str, max_size: int) -> List[str]:
        if len(text) <= max_size:
            return [text]

        sentences = re.split(r'([。！？.!?])', text)

        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence.strip():
                combined_sentences.append(sentence + punctuation)

        if not combined_sentences:
            result = []
            for i in range(0, len(text), max_size):
                result.append(text[i:i + max_size])
            return result

        segments = []
        current_segment = ""

        for sentence in combined_sentences:
            if len(sentence) > max_size:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                for i in range(0, len(sentence), max_size):
                    segments.append(sentence[i:i + max_size])
            else:
                if len(current_segment) + len(sentence) > max_size:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += sentence

        if current_segment:
            segments.append(current_segment)

        return segments

    def _safe_tokenize(self, text: str) -> List[str]:
        try:
            if len(text) > self.max_text_length:
                return self._basic_tokenize(text)

            if self.encoding:
                token_ids = self.encoding.encode(text)
                return [self.encoding.decode([token_id]) for token_id in token_ids]

            return self._basic_tokenize(text)
        except Exception:
            return self._basic_tokenize(text)

    def _basic_tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def chunk_text(self, text: str) -> List[List[str]]:
        if not text or len(text) < self.chunk_size / 10:
            tokens = self._safe_tokenize(text)
            return [tokens] if tokens else []

        text_segments = self._preprocess_large_text(text)

        all_chunks = []
        for segment in text_segments:
            segment_chunks = self._chunk_single_segment(segment)
            all_chunks.extend(segment_chunks)

        return all_chunks

    def _chunk_single_segment(self, text: str) -> List[List[str]]:
        if not text:
            return []

        all_tokens = self._safe_tokenize(text)
        if not all_tokens:
            return []

        chunks = []
        start_pos = 0

        while start_pos < len(all_tokens):
            end_pos = min(start_pos + self.chunk_size, len(all_tokens))

            if end_pos < len(all_tokens):
                sentence_end = self._find_next_sentence_end(all_tokens, end_pos)
                if sentence_end <= start_pos + self.chunk_size + 100:
                    end_pos = sentence_end

            chunk = all_tokens[start_pos:end_pos]
            if chunk:
                chunks.append(chunk)

            if end_pos >= len(all_tokens):
                break

            overlap_start = max(start_pos, end_pos - self.overlap)
            next_sentence_start = self._find_previous_sentence_end(all_tokens, overlap_start)

            if next_sentence_start > start_pos and next_sentence_start < end_pos:
                start_pos = next_sentence_start
            else:
                start_pos = overlap_start

            if start_pos >= end_pos:
                start_pos = end_pos

        return chunks

    def _is_sentence_end(self, token: str) -> bool:
        return bool(re.search(r"[.!?]\s*$", token))

    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return len(tokens)

    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        for i in range(start_pos - 1, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return 0

    def get_text_stats(self, text: str) -> dict:
        token_count = self._count_tokens(text)
        stats = {
            'text_length': len(text),
            'needs_preprocessing': len(text) > self.max_text_length,
            'estimated_chunks': max(1, token_count // self.chunk_size) if token_count else 0,
            'paragraphs': len(text.split('\n\n')),
            'lines': len(text.split('\n'))
        }

        if stats['needs_preprocessing']:
            segments = self._preprocess_large_text(text)
            stats['preprocessed_segments'] = len(segments)
            stats['max_segment_length'] = max(len(seg) for seg in segments) if segments else 0

        return stats

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                return len(self._basic_tokenize(text))
        return len(self._basic_tokenize(text))
