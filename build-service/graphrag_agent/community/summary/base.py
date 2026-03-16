from abc import ABC, abstractmethod
from typing import List, Dict
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graphrag_agent.models.get_models import get_llm_model
import concurrent.futures
import time

from graphrag_agent.config.settings import MAX_WORKERS
from graphrag_agent.config.prompts import COMMUNITY_SUMMARY_PROMPT

class BaseCommunityDescriber:
    """Community information formatting utility."""

    @staticmethod
    def prepare_string(data: Dict) -> str:
        """Convert community info to a human-readable string."""
        try:
            nodes_str = "Nodes are:\n"
            for node in data.get('nodes', []):
                node_id = node.get('id', 'unknown_id')
                node_type = node.get('type', 'unknown_type')
                node_description = (
                    f", description: {node['description']}"
                    if 'description' in node and node['description']
                    else ""
                )
                nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

            rels_str = "Relationships are:\n"
            for rel in data.get('rels', []):
                start = rel.get('start', 'unknown_start')
                end = rel.get('end', 'unknown_end')
                rel_type = rel.get('type', 'unknown_type')
                description = (
                    f", description: {rel['description']}"
                    if 'description' in rel and rel['description']
                    else ""
                )
                rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

            return nodes_str + "\n" + rels_str
        except Exception as e:
            print(f"Error formatting community info: {e}")
            return f"Error: {str(e)}\nData: {str(data)}"

class BaseCommunityRanker:
    """Community rank computation utility."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def calculate_ranks(self) -> None:
        """Compute community ranks."""
        start_time = time.time()
        print("Computing community ranks...")

        try:
            result = self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(:`__Entity__`)<-[:MENTIONS]-(d:`__Chunk__`)
            WITH c, count(distinct d) AS rank
            SET c.community_rank = rank
            RETURN count(c) AS processed_count
            """)

            processed_count = result[0]['processed_count'] if result else 0
            print(f"Community rank computation complete: processed {processed_count} communities "
                  f"in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Error computing community ranks: {e}")
            self._calculate_ranks_fallback()

    def _calculate_ranks_fallback(self):
        """Fallback rank computation method."""
        try:
            self.graph.query("""
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY]-(e:`__Entity__`)
            WITH c, count(e) AS entity_count
            SET c.community_rank = entity_count
            """)
            print("Using entity count as community rank")
        except Exception as e:
            print(f"Fallback rank computation also failed: {e}")

class BaseCommunityStorer:
    """Community information storage utility."""

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def store_summaries(self, summaries: List[Dict]) -> None:
        """Store community summaries."""
        if not summaries:
            print("No community summaries to store")
            return

        start_time = time.time()
        print(f"Storing {len(summaries)} community summaries...")

        batch_size = min(100, max(10, len(summaries) // 5))
        total_batches = (len(summaries) + batch_size - 1) // batch_size

        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            batch_start = time.time()

            try:
                self.graph.query("""
                UNWIND $data AS row
                MERGE (c:__Community__ {id:row.community})
                SET c.summary = row.summary,
                    c.full_content = row.full_content,
                    c.summary_created_at = datetime()
                """, params={"data": batch})

                print(f"Stored batch {i//batch_size + 1}/{total_batches} "
                      f"in {time.time() - batch_start:.2f}s")

            except Exception as e:
                print(f"Error storing community summary batch: {e}")
                self._store_summaries_one_by_one(batch)

    def _store_summaries_one_by_one(self, summaries: List[Dict]):
        """Store community summaries one by one."""
        for summary in summaries:
            try:
                self.graph.query("""
                MERGE (c:__Community__ {id:$community})
                SET c.summary = $summary,
                    c.full_content = $full_content,
                    c.summary_created_at = datetime()
                """, params=summary)
            except Exception as e:
                print(f"Error storing individual community summary: {e}")

class BaseSummarizer(ABC):
    """Base class for community summarizers."""

    def __init__(self, graph: Neo4jGraph):
        """Initialize the base community summarizer."""
        self.graph = graph
        self.llm = get_llm_model()
        self.describer = BaseCommunityDescriber()
        self.ranker = BaseCommunityRanker(graph)
        self.storer = BaseCommunityStorer(graph)
        self._setup_llm_chain()

        # Performance monitoring
        self.llm_time = 0
        self.query_time = 0
        self.store_time = 0

        self.max_workers = MAX_WORKERS
        print(f"Community summarizer initialized with {self.max_workers} parallel threads")

    def _setup_llm_chain(self) -> None:
        """Set up the LLM processing chain."""
        try:
            community_prompt = ChatPromptTemplate.from_messages([
                ("system", COMMUNITY_SUMMARY_PROMPT),
                ("human", "{community_info}"),
            ])
            self.community_chain = community_prompt | self.llm | StrOutputParser()
        except Exception as e:
            print(f"Error setting up LLM processing chain: {e}")
            raise

    @abstractmethod
    def collect_community_info(self) -> List[Dict]:
        """Abstract method to collect community info."""
        pass

    def process_communities(self) -> List[Dict]:
        """Process all communities."""
        total_start_time = time.time()
        print("Starting community summary processing...")

        try:
            # Compute community ranks
            rank_start = time.time()
            self.ranker.calculate_ranks()
            rank_time = time.time() - rank_start

            # Collect community info
            query_start = time.time()
            community_info = self.collect_community_info()
            self.query_time = time.time() - query_start

            if not community_info:
                print("No communities found to process")
                return []

            # Generate summaries in parallel
            llm_start = time.time()
            optimal_workers = min(self.max_workers, max(1, len(community_info) // 2))
            print(f"Generating {len(community_info)} community summaries in parallel "
                  f"using {optimal_workers} threads...")

            summaries = self._process_communities_parallel(
                community_info,
                optimal_workers
            )

            self.llm_time = time.time() - llm_start

            # Save summaries
            store_start = time.time()
            self.storer.store_summaries(summaries)
            self.store_time = time.time() - store_start

            # Output performance statistics
            total_time = time.time() - total_start_time
            self._print_performance_stats(
                total_time, rank_time,
                self.query_time, self.llm_time,
                self.store_time
            )

            return summaries

        except Exception as e:
            print(f"Error processing community summaries: {str(e)}")
            raise

    def _process_communities_parallel(
        self,
        community_info: List[Dict],
        workers: int
    ) -> List[Dict]:
        """Process community summaries in parallel."""
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_community = {
                executor.submit(self._process_single_community, info): i
                for i, info in enumerate(community_info)
            }

            for i, future in enumerate(concurrent.futures.as_completed(future_to_community)):
                try:
                    result = future.result()
                    summaries.append(result)

                    if (i+1) % 10 == 0 or (i+1) == len(community_info):
                        print(f"Processed {i+1}/{len(community_info)} "
                              f"({(i+1)/len(community_info)*100:.1f}%)")

                except Exception as e:
                    print(f"Error processing community summary: {e}")

        return summaries

    def _process_single_community(self, community: Dict) -> Dict:
        """Process a single community summary."""
        community_id = community.get('communityId', 'unknown')

        try:
            stringify_info = self.describer.prepare_string(community)

            if len(stringify_info) < 10:
                print(f"Community {community_id} has too little information, skipping summary")
                return {
                    "community": community_id,
                    "summary": "This community does not have enough information to generate a summary.",
                    "full_content": stringify_info
                }

            summary = self.community_chain.invoke({'community_info': stringify_info})

            return {
                "community": community_id,
                "summary": summary,
                "full_content": stringify_info
            }
        except Exception as e:
            print(f"Error processing summary for community {community_id}: {e}")
            return {
                "community": community_id,
                "summary": f"Error generating summary: {str(e)}",
                "full_content": str(community)
            }

    def _print_performance_stats(
        self,
        total_time: float,
        rank_time: float,
        query_time: float,
        llm_time: float,
        store_time: float
    ) -> None:
        """Print performance statistics."""
        print(f"\nCommunity summary processing complete, total time: {total_time:.2f}s")
        print(f"  Community rank computation: {rank_time:.2f}s ({rank_time/total_time*100:.1f}%)")
        print(f"  Community info query: {query_time:.2f}s ({query_time/total_time*100:.1f}%)")
        print(f"  Summary generation (LLM): {llm_time:.2f}s ({llm_time/total_time*100:.1f}%)")
        print(f"  Result storage: {store_time:.2f}s ({store_time/total_time*100:.1f}%)")
