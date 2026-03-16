import time
from typing import List, Dict, Any, Optional

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.config.settings import (
    ALIGNMENT_CONFLICT_THRESHOLD,
    ALIGNMENT_MIN_GROUP_SIZE
)
from graphrag_agent.config.prompts import entity_alignment_prompt

class EntityAligner:
    """
    Entity aligner: canonical_id grouping → conflict detection → merge.

    Aligns and merges entities that share the same canonical_id, resolving conflicts.
    """

    def __init__(self):
        self.graph = connection_manager.get_connection()
        self.llm = get_llm_model()

        # Performance statistics
        self.stats = {
            'groups_found': 0,
            'conflicts_detected': 0,
            'entities_aligned': 0
        }

    def group_by_canonical_id(self, skip: int = 0, limit: int = 100) -> Dict[str, List[str]]:
        """
        Phase 1: Group by canonical_id.
        Find all entities pointing to the same canonical_id.
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.canonical_id IS NOT NULL
        WITH e.canonical_id AS canonical_id, collect(e.id) AS entity_ids
        WHERE size(entity_ids) >= $min_size
        RETURN canonical_id, entity_ids
        ORDER BY size(entity_ids) DESC
        SKIP $skip
        LIMIT $limit
        """

        results = self.graph.query(query, params={
            'min_size': ALIGNMENT_MIN_GROUP_SIZE,
            'skip': skip,
            'limit': limit
        })

        groups = {}
        for row in results:
            canonical_id = row['canonical_id']
            entity_ids = row['entity_ids']
            # Ensure canonical_id itself is included in the list
            if canonical_id not in entity_ids:
                entity_ids.append(canonical_id)
            groups[canonical_id] = entity_ids

        return groups

    def count_alignment_groups(self) -> int:
        """
        Count the total number of groups that need alignment.
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.canonical_id IS NOT NULL
        WITH e.canonical_id AS canonical_id, collect(e.id) AS entity_ids
        WHERE size(entity_ids) >= $min_size
        RETURN count(*) AS total
        """

        result = self.graph.query(query, params={
            'min_size': ALIGNMENT_MIN_GROUP_SIZE
        })

        return result[0]['total'] if result else 0

    def detect_conflicts(self, canonical_id: str, entity_ids: List[str]) -> Dict[str, Any]:
        """
        Phase 2: Conflict detection.
        Detect whether entities under the same canonical_id have semantic conflicts.
        """
        # Fetch entity descriptions and relationships; use COUNT {} instead of size()
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:`__Entity__` {id: eid})
        OPTIONAL MATCH (e)-[r]->(other)
        WITH e, collect(DISTINCT type(r)) AS rel_types, count(r) AS rel_count
        RETURN e.id AS entity_id,
               e.description AS description,
               rel_types,
               rel_count
        """

        entities = self.graph.query(query, params={'entity_ids': entity_ids})

        # Simple conflict detection: if relationship type sets differ too much, there may be a conflict
        if len(entities) < 2:
            return {'has_conflict': False, 'entities': entities}

        # Compute the intersection ratio of relationship type sets
        all_rel_types = [set(e['rel_types']) for e in entities if e['rel_types']]
        if all_rel_types:
            intersection = set.intersection(*all_rel_types) if len(all_rel_types) > 1 else all_rel_types[0]
            union = set.union(*all_rel_types)

            jaccard = len(intersection) / len(union) if union else 0

            has_conflict = jaccard < ALIGNMENT_CONFLICT_THRESHOLD

            if has_conflict:
                self.stats['conflicts_detected'] += 1

            return {
                'has_conflict': has_conflict,
                'jaccard_similarity': jaccard,
                'entities': entities
            }

        return {'has_conflict': False, 'entities': entities}

    def resolve_conflict(self, canonical_id: str, conflict_info: Dict[str, Any]) -> str:
        """
        Use an LLM to resolve conflicts and decide which entity to keep.
        """
        entities = conflict_info['entities']

        # Build the LLM prompt
        entity_desc = "\n".join([
            f"- {e['entity_id']}: {e['description']}, {e['rel_count']} relations"
            for e in entities
        ])

        prompt = entity_alignment_prompt.format(entity_desc=entity_desc)

        try:
            response = self.llm.invoke(prompt)
            selected = response.content.strip()

            # Validate that the selected ID is in the list
            valid_ids = [e['entity_id'] for e in entities]
            if selected in valid_ids:
                return selected
        except:
            pass

        # Fallback: select the entity with the most relationships
        return max(entities, key=lambda x: x['rel_count'])['entity_id']

    def merge_entities(self, canonical_id: str, entity_ids: List[str], keep_id: Optional[str] = None) -> int:
        """
        Phase 3: Merge entities.
        Merge all entities into the canonical entity, keeping only one.
        Preserves original relationship types to avoid losing semantic information.

        Uses CALL subqueries to isolate edge processing, ensuring that even when
        there are no edges the main flow can still execute SET and DELETE.
        """
        if not entity_ids or len(entity_ids) < 2:
            return 0

        # Determine which entity to keep
        target_id = keep_id or canonical_id

        # Ensure target is in entity_ids
        if target_id not in entity_ids:
            target_id = entity_ids[0]

        # Entities to delete
        to_delete = [eid for eid in entity_ids if eid != target_id]

        if not to_delete:
            return 0

        # Merge query: use CALL subqueries to isolate edge processing
        merge_query = """
        // 1. Ensure target entity exists
        MERGE (target:`__Entity__` {id: $target_id})

        WITH target, size($to_delete) AS deletion_count

        // 2. Process each entity to delete
        UNWIND $to_delete AS del_id
        MATCH (old:`__Entity__` {id: del_id})

        // 3. Handle outgoing edges in a subquery (does not affect the main flow)
        CALL {
            WITH old, target
            // Collect outgoing edge info
            OPTIONAL MATCH (old)-[r_out]->(other)
            WHERE other.id <> $target_id
            WITH old, target,
                type(r_out) AS rel_type,
                other,
                properties(r_out) AS rel_props
            WHERE rel_type IS NOT NULL AND other IS NOT NULL

            // Check whether the target already has a relationship of the same type to this node
            OPTIONAL MATCH (target)-[existing]->(other)
            WHERE type(existing) = rel_type

            WITH old, target, rel_type, other, rel_props,
                 collect(properties(existing)) AS existing_props
            // Only create if an identical relationship (by type and properties) doesn't already exist
            WHERE NOT rel_props IN existing_props

            CALL apoc.create.relationship(target, rel_type, rel_props, other)
            YIELD rel
            RETURN count(rel) AS out_edges_created
        }

        // 4. Handle incoming edges in a subquery (does not affect the main flow)
        WITH old, target, deletion_count, out_edges_created
        CALL {
            WITH old, target
            // Collect incoming edge info
            OPTIONAL MATCH (other)-[r_in]->(old)
            WHERE other.id <> $target_id
            WITH old, target,
                type(r_in) AS rel_type,
                other,
                properties(r_in) AS rel_props
            WHERE rel_type IS NOT NULL AND other IS NOT NULL

            // Check whether a relationship of the same type already exists from this node to target
            OPTIONAL MATCH (other)-[existing]->(target)
            WHERE type(existing) = rel_type

            WITH old, target, rel_type, other, rel_props,
                 collect(properties(existing)) AS existing_props
            // Only create if an identical relationship doesn't already exist
            WHERE NOT rel_props IN existing_props

            CALL apoc.create.relationship(other, rel_type, rel_props, target)
            YIELD rel
            RETURN count(rel) AS in_edges_created
        }

        // 5. Merge properties and mark (always executes, independent of edge processing)
        WITH target, old, deletion_count, out_edges_created, in_edges_created
        SET target.description = COALESCE(target.description, old.description),
            target.aligned_from = COALESCE(target.aligned_from, []) + [old.id],
            target.aligned_at = datetime(),
            target.canonical_id = $target_id

        // 6. Delete the old entity (always executes)
        DETACH DELETE old

        RETURN deletion_count AS deleted,
            sum(out_edges_created) AS total_out_edges,
            sum(in_edges_created) AS total_in_edges
        """

        try:
            result = self.graph.query(merge_query, params={
                'target_id': target_id,
                'to_delete': to_delete
            })

            if result and len(result) > 0:
                deleted = result[0].get('deleted', 0)
                out_edges = result[0].get('total_out_edges', 0)
                in_edges = result[0].get('total_in_edges', 0)

                self.stats['entities_aligned'] += deleted

                if deleted > 0:
                    print(f"Merge successful: deleted {deleted} entities, transferred {out_edges} outgoing and {in_edges} incoming edges")

                return deleted
            else:
                print(f"Warning: merge query returned empty result, target={target_id}, to_delete={to_delete}")
                return 0

        except Exception as e:
            print(f"Error merging entities: {e}, target={target_id}, to_delete={to_delete}")
            # Do not raise — return 0 and continue processing other groups
            return 0

    def align_all(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Execute the full alignment pipeline.
        """
        start_time = time.time()

        # Count total groups
        total_groups = self.count_alignment_groups()
        print(f"Found {total_groups} canonical groups to align")

        self.stats['groups_found'] = total_groups
        total_merged = 0
        groups_processed = 0
        skip = 0
        while True:
            # Fetch the current batch
            groups = self.group_by_canonical_id(skip=skip, limit=batch_size)

            if not groups:
                # No more groups — exit loop
                break

            batch_count = len(groups)
            print(f"Processing batch: skip={skip}, fetched {batch_count} groups")

            # Process each group in the current batch
            for canonical_id, entity_ids in groups.items():
                # Conflict detection
                conflict_info = self.detect_conflicts(canonical_id, entity_ids)

                if conflict_info['has_conflict']:
                    # Resolve the conflict
                    keep_id = self.resolve_conflict(canonical_id, conflict_info)
                else:
                    keep_id = canonical_id

                # Merge entities
                merged = self.merge_entities(canonical_id, entity_ids, keep_id)
                total_merged += merged
                groups_processed += 1

            # Advance to the next batch
            skip += batch_size

            # If the current batch is smaller than batch_size, this is the last batch
            if batch_count < batch_size:
                break

        elapsed = time.time() - start_time

        return {
            'groups_processed': groups_processed,
            'entities_aligned': total_merged,
            'conflicts_detected': self.stats['conflicts_detected'],
            'elapsed_time': elapsed
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.stats
