import time
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
import numpy as np

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.config.settings import (
    DISAMBIG_STRING_THRESHOLD,
    DISAMBIG_VECTOR_THRESHOLD,
    DISAMBIG_NIL_THRESHOLD,
    DISAMBIG_TOP_K
)

class EntityDisambiguator:
    """
    Entity disambiguator: mention → string recall → vector reranking → NIL detection → canonical_id

    Eliminates entity ambiguity through a multi-stage pipeline, mapping mentions to
    canonical entities in the knowledge graph.
    """

    def __init__(self):
        self.graph = connection_manager.get_connection()
        self.embeddings = get_embeddings_model()

        # Performance statistics
        self.stats = {
            'mentions_processed': 0,
            'candidates_recalled': 0,
            'nil_detected': 0,
            'disambiguated': 0
        }

    def string_recall(self, mention: str, top_k: int = DISAMBIG_TOP_K) -> List[Dict[str, Any]]:
        """
        Phase 1: String recall of candidate entities.
        Uses edit distance and fuzzy matching to quickly recall similar entities.
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.id IS NOT NULL
        WITH e,
             apoc.text.levenshteinSimilarity(toLower($mention), toLower(e.id)) AS similarity
        WHERE similarity >= $threshold
        RETURN e.id AS entity_id,
               e.description AS description,
               similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """

        results = self.graph.query(query, params={
            'mention': mention,
            'threshold': DISAMBIG_STRING_THRESHOLD,
            'top_k': top_k
        })

        self.stats['candidates_recalled'] += len(results)
        return results

    def vector_rerank(self, mention: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phase 2: Vector reranking of candidate entities.
        Re-orders candidates using semantic similarity.
        """
        if not candidates:
            return []

        # Compute embedding for the mention
        mention_vec = self.embeddings.embed_query(mention)

        # Fetch embeddings for candidate entities
        entity_ids = [c['entity_id'] for c in candidates]
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:`__Entity__` {id: eid})
        WHERE e.embedding IS NOT NULL
        RETURN e.id AS entity_id, e.embedding AS embedding
        """

        embeddings_result = self.graph.query(query, params={'entity_ids': entity_ids})
        embeddings_map = {r['entity_id']: r['embedding'] for r in embeddings_result}

        # Compute vector similarity and rerank
        reranked = []
        for candidate in candidates:
            entity_id = candidate['entity_id']
            if entity_id in embeddings_map:
                entity_vec = embeddings_map[entity_id]
                similarity = self._cosine_similarity(mention_vec, entity_vec)

                reranked.append({
                    **candidate,
                    'vector_similarity': similarity,
                    'combined_score': 0.4 * candidate['similarity'] + 0.6 * similarity
                })

        return sorted(reranked, key=lambda x: x['combined_score'], reverse=True)

    def nil_detection(self, mention: str, candidates: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Phase 3: NIL detection.
        Determines whether the mention is an out-of-knowledge-base entity (not in the KB).
        """
        if not candidates:
            return True, None

        # Check the score of the best candidate
        best_candidate = candidates[0]
        if best_candidate.get('combined_score', 0) < DISAMBIG_NIL_THRESHOLD:
            self.stats['nil_detected'] += 1
            return True, None

        return False, best_candidate['entity_id']

    def disambiguate(self, mention: str) -> Dict[str, Any]:
        """
        Run the full disambiguation pipeline.
        """
        self.stats['mentions_processed'] += 1

        # String recall
        candidates = self.string_recall(mention)

        if not candidates:
            return {
                'mention': mention,
                'canonical_id': None,
                'is_nil': True,
                'candidates': []
            }

        # Vector reranking
        reranked = self.vector_rerank(mention, candidates)

        # NIL detection
        is_nil, canonical_id = self.nil_detection(mention, reranked)

        if not is_nil:
            self.stats['disambiguated'] += 1

        return {
            'mention': mention,
            'canonical_id': canonical_id,
            'is_nil': is_nil,
            'candidates': reranked[:3]  # Return top 3 candidates
        }

    def batch_disambiguate(self, mentions: List[str]) -> List[Dict[str, Any]]:
        """Batch disambiguation."""
        results = []
        for mention in mentions:
            result = self.disambiguate(mention)
            results.append(result)

        return results

    def apply_to_graph(self) -> int:
        """
        Apply disambiguation results to the graph.
        Core idea: find merged entity groups and set canonical_id on non-canonical entities
        pointing to the primary entity in each group.

        Pagination strategy:
        - Each iteration queries the first batch_size unprocessed groups (WHERE canonical_id IS NULL)
        - After processing, those groups disappear; the next query automatically returns new ones
        - Loop until the query returns an empty result set
        - No SKIP needed because the result set shrinks automatically
        """
        total_updated = 0
        batch_size = 500
        processed_groups = 0
        iteration = 0

        print(f"Starting paginated WCC group processing, batch size: {batch_size}")
        print(f"Strategy: query first {batch_size} unprocessed groups per round; processed groups are removed automatically")

        while True:
            iteration += 1

            # Query the first batch_size unprocessed groups each round
            # WHERE canonical_id IS NULL ensures only unprocessed groups are returned
            # No SKIP — always start from the beginning
            query = """
            MATCH (e:`__Entity__`)
            WHERE e.wcc IS NOT NULL
            AND e.embedding IS NOT NULL
            AND e.canonical_id IS NULL
            WITH e.wcc AS community, collect(e) AS entities
            WHERE size(entities) >= 2
            WITH community, entities
            ORDER BY community
            LIMIT $limit
            UNWIND entities AS entity
            WITH community, entity, COUNT { (entity)--() } AS degree
            WITH community, collect({id: entity.id, description: entity.description, degree: degree}) AS entity_info
            RETURN community, entity_info
            """

            # Note: params contain only limit, no skip
            groups = self.graph.query(query, params={'limit': batch_size})

            if not groups:
                print(f"Round {iteration}: query returned 0 groups — all data has been processed")
                break

            print(f"Round {iteration}: query returned {len(groups)} groups to process")

            # Process groups in the current batch
            batch_updated = 0
            for group in groups:
                entities = group['entity_info']

                # Select the entity with the highest degree as canonical (most representative)
                canonical = max(entities, key=lambda x: x['degree'])
                canonical_id = canonical['id']

                # Point all other entities to it
                other_ids = [e['id'] for e in entities if e['id'] != canonical_id]

                if other_ids:
                    update_query = """
                    UNWIND $entity_ids AS eid
                    MATCH (e:`__Entity__` {id: eid})
                    SET e.canonical_id = $canonical_id,
                        e.disambiguated = true,
                        e.disambiguated_at = datetime()
                    RETURN count(e) AS updated
                    """

                    result = self.graph.query(update_query, params={
                        'entity_ids': other_ids,
                        'canonical_id': canonical_id
                    })

                    if result:
                        batch_updated += result[0]['updated']

            total_updated += batch_updated
            processed_groups += len(groups)

            print(f"Round {iteration} complete: updated {batch_updated} entities")
            print(f"Cumulative: {processed_groups} groups processed, {total_updated} entities updated")

            # If this round returned fewer groups than batch_size, remaining data is less than a full batch
            # The next round will return empty and the loop will end naturally
            if len(groups) < batch_size:
                print(f"This round returned {len(groups)} < {batch_size} — remaining data is less than one batch")

        print(f"\n{'='*60}")
        print(f"Disambiguation complete:")
        print(f"  Total rounds: {iteration}")
        print(f"  WCC groups processed: {processed_groups}")
        print(f"  Entities updated: {total_updated}")
        print(f"{'='*60}\n")

        # Final validation: ensure nothing was missed
        print("Running final validation...")
        remaining_query = """
        MATCH (e:`__Entity__`)
        WHERE e.wcc IS NOT NULL
        AND e.embedding IS NOT NULL
        AND e.canonical_id IS NULL
        WITH e.wcc AS community, collect(e) AS entities
        WHERE size(entities) >= 2
        RETURN count(DISTINCT community) AS remaining_groups,
            sum(size(entities)) AS remaining_entities
        """

        remaining = self.graph.query(remaining_query)
        if remaining and remaining[0]['remaining_groups'] > 0:
            print(f"Validation failed: {remaining[0]['remaining_groups']} groups still unprocessed!")
            print(f"Contains {remaining[0]['remaining_entities']} entities")
        else:
            print(f"Validation passed: all qualifying WCC groups have been processed")

        return total_updated

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.stats
