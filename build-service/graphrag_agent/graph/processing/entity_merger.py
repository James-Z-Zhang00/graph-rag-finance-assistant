import re
import ast
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.config.prompts import system_template_build_index, user_template_build_index
from graphrag_agent.config.settings import ENTITY_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS
from graphrag_agent.graph.core import connection_manager, timer, get_performance_stats, print_performance_stats

class EntityMerger:
    """
    Entity merge manager responsible for merging similar entities based on LLM decisions.
    Main capabilities include using an LLM to analyze entity similarity, parsing merge
    suggestions, and executing entity merge operations.
    """

    def __init__(self, batch_size: int = 20, max_workers: int = 4):
        """
        Initialize the entity merge manager.

        Args:
            batch_size: Batch processing size
            max_workers: Number of parallel worker threads
        """
        # Initialize graph database connection
        self.graph = connection_manager.get_connection()

        # Get language model
        self.llm = get_llm_model()

        # Batch processing and parallelism parameters
        self.batch_size = batch_size or ENTITY_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

        # Set up LLM processing chain
        self._setup_llm_chain()

        # Create indexes
        self._create_indexes()

        # Performance monitoring
        self.llm_time = 0
        self.db_time = 0
        self.parse_time = 0

    def _create_indexes(self) -> None:
        """Create necessary indexes to optimize query performance."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)"
        ]

        connection_manager.create_multiple_indexes(index_queries)

    def _setup_llm_chain(self) -> None:
        """
        Set up the LLM processing chain for entity merge decisions.
        Creates prompt templates and builds the processing chain.
        """
        # Check model capabilities
        if not hasattr(self.llm, 'with_structured_output'):
            print("Current LLM model does not support structured output")

        # Create prompt templates
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template_build_index
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            user_template_build_index
        )

        # Build conversation chain
        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])

        # Create the final processing chain
        self.chain = self.chat_prompt | self.llm

    def _convert_to_list(self, result: str) -> List[List[str]]:
        """
        Convert LLM-returned entity list text into a Python list.

        Args:
            result: Text result returned by the LLM containing entity lists

        Returns:
            List[List[str]]: 2D list where each sub-list is a group of mergeable entities
        """
        start_time = time.time()

        # Use regex to match all bracket-enclosed content
        list_pattern = re.compile(r'\[.*?\]')
        entity_lists = []

        # First try to parse the entire result directly with ast.literal_eval
        try:
            # Find the possible start of the list
            list_start = result.find('[')
            if list_start >= 0:
                # Try to find the end of the complete list
                nested_level = 0
                for i in range(list_start, len(result)):
                    if result[i] == '[':
                        nested_level += 1
                    elif result[i] == ']':
                        nested_level -= 1
                        if nested_level == 0:
                            # Extract the potential list portion
                            list_portion = result[list_start:i+1]
                            try:
                                parsed_list = ast.literal_eval(list_portion)
                                if isinstance(parsed_list, list):
                                    # Check if it is a 2D list
                                    if all(isinstance(item, list) for item in parsed_list):
                                        entity_lists = parsed_list
                                    else:
                                        entity_lists = [parsed_list]
                                    break
                            except:
                                pass  # If parsing fails, fall through to regex method
        except:
            pass  # If above method fails, fall back to regex

        # If direct parsing failed, use regex method
        if not entity_lists:
            # Parse each matched list string
            for match in list_pattern.findall(result):
                try:
                    # Convert string to Python list
                    entity_list = ast.literal_eval(match)
                    # Only add non-empty lists
                    if entity_list and isinstance(entity_list, list):
                        if all(isinstance(item, list) for item in entity_list):
                            # If it's a nested list, extend
                            entity_lists.extend(entity_list)
                        else:
                            # If it's a single list, append
                            entity_lists.append(entity_list)
                except Exception as e:
                    print(f"Error parsing entity list: {str(e)}, raw text: {match}")

        # Filter and normalize results
        valid_lists = []
        for entity_list in entity_lists:
            # Ensure all items in the list are strings
            if all(isinstance(item, str) for item in entity_list):
                # Remove duplicates
                unique_list = list(dict.fromkeys(entity_list))
                if len(unique_list) > 1:  # Only keep groups with at least 2 entities
                    valid_lists.append(unique_list)

        self.parse_time += time.time() - start_time
        return valid_lists

    def get_merge_suggestions(self, duplicate_candidates: List[Any]) -> List[List[str]]:
        """
        Use an LLM to analyze and provide entity merge suggestions — parallel-optimized version.

        Args:
            duplicate_candidates: List of candidate duplicate entity groups

        Returns:
            List[List[str]]: List of suggested entity merge groups
        """
        # Check if there are any candidate entities
        if not duplicate_candidates:
            return []

        llm_start_time = time.time()

        # Collect LLM merge suggestions
        merged_entities = []

        # Dynamically adjust batch size
        candidate_count = len(duplicate_candidates)
        optimal_batch_size = min(self.max_workers * 2, max(1, candidate_count // 5))

        print(f"Processing {candidate_count} candidate entity groups, batch size: {optimal_batch_size}")

        # Process in batches to avoid creating too many threads
        for batch_start in range(0, candidate_count, optimal_batch_size):
            batch_end = min(batch_start + optimal_batch_size, candidate_count)
            batch = duplicate_candidates[batch_start:batch_end]

            # Use thread pool for parallel LLM requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_candidates = {
                    executor.submit(self._process_candidate_group, candidates): i
                    for i, candidates in enumerate(batch)
                }

                # Collect results
                for future in concurrent.futures.as_completed(future_to_candidates):
                    try:
                        result = future.result()
                        if result:
                            merged_entities.append(result)
                    except Exception as e:
                        print(f"Error processing candidate entity group: {e}")

            # Report progress
            print(f"Processed {batch_end}/{candidate_count} candidate entity groups")

        self.llm_time += time.time() - llm_start_time

        parse_start_time = time.time()
        # Parse and organize the final merge suggestions
        results = []
        for candidates in merged_entities:
            # Convert each suggestion to list format
            temp = self._convert_to_list(candidates)
            results.extend(temp)

        self.parse_time += time.time() - parse_start_time

        # Merge groups that share entities
        merged_results = self._merge_overlapping_groups(results)

        print(f"LLM analysis complete: found {len(merged_results)} mergeable entity groups")
        return merged_results

    def _merge_overlapping_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """
        Merge overlapping entity groups.

        Args:
            groups: List of entity groups

        Returns:
            List[List[str]]: Merged entity groups
        """
        if not groups:
            return []

        # Create entity-to-group mapping
        entity_to_groups = {}
        for i, group in enumerate(groups):
            for entity in group:
                if entity not in entity_to_groups:
                    entity_to_groups[entity] = []
                entity_to_groups[entity].append(i)

        # Use union-find to merge connected groups
        parent = list(range(len(groups)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        # Merge groups that share at least one common entity
        for entity, group_indices in entity_to_groups.items():
            if len(group_indices) > 1:
                for i in range(1, len(group_indices)):
                    union(group_indices[0], group_indices[i])

        # Collect merged groups
        merged_groups = {}
        for i, group in enumerate(groups):
            root = find(i)
            if root not in merged_groups:
                merged_groups[root] = set()
            merged_groups[root].update(group)

        # Convert back to list format
        return [list(entities) for entities in merged_groups.values()]

    def _process_candidate_group(self, candidates: List[str]) -> Optional[str]:
        """
        Process a single candidate entity group.

        Args:
            candidates: List of candidate entities

        Returns:
            str: LLM analysis result
        """
        if not candidates or len(candidates) < 2:
            return None

        chat_history = []
        max_retries = 2
        for retry in range(max_retries + 1):
            try:
                # Call LLM for analysis
                answer = self.chain.invoke({
                    "chat_history": chat_history,
                    "entities": candidates
                })
                return answer.content
            except Exception as e:
                if retry < max_retries:
                    print(f"LLM call failed, retrying ({retry+1}/{max_retries}): {e}")
                    time.sleep(1)  # Brief delay
                else:
                    print(f"LLM call failed after max retries: {e}")
                    return None

    def execute_merges(self, merge_groups: List[List[str]]) -> int:
        """
        Execute entity merge operations — batch-optimized version.

        Args:
            merge_groups: List of entity groups to merge

        Returns:
            int: Number of nodes affected by merge operations
        """
        if not merge_groups:
            return 0

        db_start_time = time.time()

        # Dynamic batch size
        group_count = len(merge_groups)
        optimal_batch_size = min(self.batch_size, max(5, group_count // 10))
        total_batches = (group_count + optimal_batch_size - 1) // optimal_batch_size

        print(f"Starting {group_count} entity merge groups, batch size: {optimal_batch_size}")

        total_merged = 0
        batch_times = []

        # Process merges in batches
        for batch_index in range(total_batches):
            batch_start = time.time()

            start_idx = batch_index * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, group_count)
            batch = merge_groups[start_idx:end_idx]

            try:
                # Execute Neo4j merge operation
                result = self.graph.query("""
                UNWIND $data AS candidates
                CALL {
                  WITH candidates
                  MATCH (e:__Entity__) WHERE e.id IN candidates
                  RETURN collect(e) AS nodes
                }
                CALL apoc.refactor.mergeNodes(nodes, {properties: {
                    `.*`: 'discard'
                }})
                YIELD node
                RETURN count(*) as merged_count
                """, params={"data": batch})

                if result:
                    batch_merged = result[0]["merged_count"]
                    total_merged += batch_merged

                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    batch_times.append(batch_time)

                    # Calculate average time and estimated remaining time
                    avg_time = sum(batch_times) / len(batch_times)
                    remaining_batches = total_batches - (batch_index + 1)
                    estimated_remaining = avg_time * remaining_batches

                    print(f"Merge batch {batch_index+1}/{total_batches}: "
                          f"merged {batch_merged} entities, "
                          f"batch time: {batch_time:.2f}s, "
                          f"estimated remaining: {estimated_remaining:.2f}s")
            except Exception as e:
                print(f"Batch merge failed, falling back to individual processing: {e}")
                batch_merged = 0

                # If batch processing fails, try one group at a time
                for group in batch:
                    try:
                        single_result = self.graph.query("""
                        MATCH (e:__Entity__) WHERE e.id IN $candidates
                        WITH collect(e) AS nodes
                        CALL apoc.refactor.mergeNodes(nodes, {properties: {
                            `.*`: 'discard'
                        }})
                        YIELD node
                        RETURN count(*) as merged_count
                        """, params={"candidates": group})

                        if single_result:
                            group_merged = single_result[0]["merged_count"]
                            total_merged += group_merged
                            batch_merged += group_merged
                    except Exception as e2:
                        print(f"Single group merge failed: {e2}")

                print(f"Individual processing complete for this batch: merged {batch_merged} entities")

        self.db_time += time.time() - db_start_time

        return total_merged

    def clean_duplicate_relationships(self):
        """
        Remove duplicate relationships, including:
        1. Duplicate relationships in the same direction
        2. Bidirectional redundancy in SIMILAR relationships (keep one direction)
        """
        print("Starting duplicate relationship cleanup...")

        # Step 1: Remove duplicate relationships in the same direction
        result1 = self.graph.query("""
        MATCH (a)-[r]->(b)
        WITH a, b, type(r) as type, collect(r) as rels
        WHERE size(rels) > 1
        WITH a, b, type, rels[0] as kept, rels[1..] as rels
        UNWIND rels as rel
        DELETE rel
        RETURN count(*) as deleted
        """)

        deleted_count1 = result1[0]["deleted"] if result1 else 0
        print(f"Deleted {deleted_count1} duplicate same-direction relationships")

        # Step 2: Remove bidirectional SIMILAR relationship redundancy (keep one direction)
        result2 = self.graph.query("""
        // Find all bidirectional SIMILAR relationships
        MATCH (a)-[r1:SIMILAR]->(b)
        MATCH (b)-[r2:SIMILAR]->(a)
        WHERE a.id < b.id  // Ensure each node pair is processed only once

        // Delete one direction (b->a)
        DELETE r2

        RETURN count(*) as deleted_bidirectional
        """)

        deleted_count2 = result2[0]["deleted_bidirectional"] if result2 else 0
        print(f"Deleted {deleted_count2} redundant bidirectional SIMILAR relationships")

        total_deleted = deleted_count1 + deleted_count2
        print(f"Total duplicate relationships removed: {total_deleted}")

        return total_deleted

    @timer
    def process_duplicates(self, duplicate_candidates: List[Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Full pipeline for processing duplicate entities: get merge suggestions and execute merges
        — performance-optimized version.

        Args:
            duplicate_candidates: List of candidate duplicate entity groups

        Returns:
            Tuple[int, Dict[str, Any]]: Number of merged entities and performance statistics
        """
        start_time = time.time()

        # Normalize duplicate_candidates to a list of lists, handling different data structures
        fixed_candidates = []
        for candidates in duplicate_candidates:
            # Check if the candidate group is a dict-like object
            if isinstance(candidates, dict) and "combinedResult" in candidates:
                candidate_list = candidates["combinedResult"]
                if isinstance(candidate_list, list) and len(candidate_list) > 1:
                    fixed_candidates.append(candidate_list)
            # Check if the candidate group is already a list
            elif isinstance(candidates, list) and len(candidates) > 1:
                fixed_candidates.append(candidates)

        # Filter out groups with too few candidates
        filtered_candidates = [
            candidates for candidates in fixed_candidates
            if len(candidates) > 1
        ]

        print(f"Candidate entity groups after filtering: {len(filtered_candidates)}")
        print(f"Processing {len(filtered_candidates)} valid duplicate candidate groups...")

        # Get merge suggestions
        merge_groups = self.get_merge_suggestions(filtered_candidates)

        suggestion_time = time.time()
        suggestion_elapsed = suggestion_time - start_time
        print(f"Merge suggestions generated in {suggestion_elapsed:.2f}s: "
            f"found {len(merge_groups)} mergeable entity groups")
        print(f"  LLM processing time: {self.llm_time:.2f}s, parse time: {self.parse_time:.2f}s")

        # Execute merges if there are suggested groups
        merged_count = 0
        if merge_groups:
            merged_count = self.execute_merges(merge_groups)

        # Clean up duplicate relationships
        self.clean_duplicate_relationships()

        end_time = time.time()
        merge_elapsed = end_time - suggestion_time
        total_elapsed = end_time - start_time

        print(f"Entity merge complete in {merge_elapsed:.2f}s: merged {merged_count} entities")
        print(f"Database operation time: {self.db_time:.2f}s")
        print(f"Total elapsed: {total_elapsed:.2f}s")

        # Return performance statistics summary
        time_records = {
            "LLM processing time": self.llm_time,
            "Parse time": self.parse_time,
            "Database time": self.db_time
        }

        performance_stats = get_performance_stats(total_elapsed, time_records)
        performance_stats.update({
            "Candidate entity groups": len(filtered_candidates),
            "Merge groups identified": len(merge_groups),
            "Entities merged": merged_count
        })

        print_performance_stats(performance_stats)

        return merged_count, performance_stats
