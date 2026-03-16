"""
Batch Neo4j queries for source/content lookups.
Ported from the monolith server/utils/neo4j_batch.py.
"""

import os
from typing import List, Dict
from neo4j import Result


class BatchProcessor:

    @staticmethod
    def get_source_info_batch(source_ids: List[str], driver) -> Dict[str, Dict]:
        if not source_ids:
            return {}

        source_info = {}
        chunk_ids = []
        community_ids = []

        for source_id in source_ids:
            if not source_id:
                source_info[source_id] = {"file_name": "Unknown file"}
                continue
            if len(source_id) == 40:
                chunk_ids.append(source_id)
            else:
                id_parts = source_id.split(",")
                if len(id_parts) >= 2 and id_parts[0] == "2":
                    chunk_ids.append(id_parts[-1])
                else:
                    community_id = id_parts[1] if len(id_parts) > 1 else source_id
                    community_ids.append(community_id)

        try:
            if chunk_ids:
                chunk_results = driver.execute_query(
                    "MATCH (n:__Chunk__) WHERE n.id IN $ids RETURN n.id AS id, n.fileName AS fileName",
                    {"ids": chunk_ids},
                    result_transformer_=Result.to_df,
                )
                if not chunk_results.empty:
                    for _, row in chunk_results.iterrows():
                        chunk_id = row["id"]
                        file_name = row["fileName"]
                        base_name = os.path.basename(file_name) if file_name else "未知文件"
                        for src_id in source_ids:
                            if chunk_id == src_id or (
                                len(src_id.split(",")) >= 2 and src_id.split(",")[-1] == chunk_id
                            ):
                                source_info[src_id] = {"file_name": base_name}

            if community_ids:
                community_results = driver.execute_query(
                    "MATCH (n:__Community__) WHERE n.id IN $ids RETURN n.id AS id",
                    {"ids": community_ids},
                    result_transformer_=Result.to_df,
                )
                if not community_results.empty:
                    for _, row in community_results.iterrows():
                        community_id = row["id"]
                        for src_id in source_ids:
                            id_parts = src_id.split(",")
                            if (len(id_parts) > 1 and id_parts[1] == community_id) or src_id == community_id:
                                source_info[src_id] = {"file_name": "Community summary"}

            for source_id in source_ids:
                if source_id not in source_info:
                    source_info[source_id] = {"file_name": f"Source text {source_id}"}

            return source_info

        except Exception as e:
            print(f"Failed to get source info batch: {e}")
            return {sid: {"file_name": f"Source text {sid}"} for sid in source_ids}

    @staticmethod
    def get_content_batch(chunk_ids: List[str], driver) -> Dict[str, Dict]:
        if not chunk_ids:
            return {}

        chunk_content = {}
        direct_chunk_ids = []
        community_ids = []

        for chunk_id in chunk_ids:
            if not chunk_id:
                chunk_content[chunk_id] = {"content": "No valid source ID provided"}
                continue
            if len(chunk_id) == 40:
                direct_chunk_ids.append(chunk_id)
            else:
                id_parts = chunk_id.split(",")
                if len(id_parts) >= 2 and id_parts[0] == "2":
                    direct_chunk_ids.append(id_parts[-1])
                else:
                    community_id = id_parts[1] if len(id_parts) > 1 else chunk_id
                    community_ids.append(community_id)

        try:
            if direct_chunk_ids:
                chunk_results = driver.execute_query(
                    "MATCH (n:__Chunk__) WHERE n.id IN $ids RETURN n.id AS id, n.fileName AS fileName, n.text AS text",
                    {"ids": direct_chunk_ids},
                    result_transformer_=Result.to_df,
                )
                if not chunk_results.empty:
                    for _, row in chunk_results.iterrows():
                        cid = row["id"]
                        file_name = row.get("fileName", "未知文件")
                        text = row.get("text", "")
                        content = f"File name: {file_name}\n\n{text}"
                        for original_id in chunk_ids:
                            if cid == original_id or (
                                len(original_id.split(",")) >= 2 and original_id.split(",")[-1] == cid
                            ):
                                chunk_content[original_id] = {"content": content}

            if community_ids:
                community_results = driver.execute_query(
                    "MATCH (n:__Community__) WHERE n.id IN $ids RETURN n.id AS id, n.summary AS summary, n.full_content AS full_content",
                    {"ids": community_ids},
                    result_transformer_=Result.to_df,
                )
                if not community_results.empty:
                    for _, row in community_results.iterrows():
                        comm_id = row["id"]
                        summary = row.get("summary", "")
                        full_content = row.get("full_content", "")
                        content = f"Summary:\n{summary}\n\nFull content:\n{full_content}"
                        for original_id in chunk_ids:
                            id_parts = original_id.split(",")
                            if (len(id_parts) > 1 and id_parts[1] == comm_id) or original_id == comm_id:
                                chunk_content[original_id] = {"content": content}

            for chunk_id in chunk_ids:
                if chunk_id not in chunk_content:
                    chunk_content[chunk_id] = {"content": f"No relevant content found: source ID {chunk_id}"}

            return chunk_content

        except Exception as e:
            print(f"Failed to get content batch: {e}")
            return {cid: {"content": f"Error occurred while retrieving source content: {str(e)}", "chunk_id": cid} for cid in chunk_ids}
