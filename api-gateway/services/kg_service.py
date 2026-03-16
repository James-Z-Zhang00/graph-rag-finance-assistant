"""
Knowledge-graph service functions.
Ported from the monolith server/services/kg_service.py.
All Neo4j access goes through db.connection.
"""

import os
import re
import traceback
from typing import Dict, List, Any
from neo4j import Result

from db.connection import get_db_manager

db_manager = get_db_manager()
driver = db_manager.driver


# ---------------------------------------------------------------------------
# KG extraction helpers
# ---------------------------------------------------------------------------

def extract_kg_from_message(message: str, query: str = None, reference: Dict = None) -> Dict:
    try:
        if reference and isinstance(reference, dict):
            chunks = reference.get("chunks", [])
            chunk_ids = reference.get("Chunks", [])
            for chunk in chunks:
                if "chunk_id" in chunk:
                    chunk_ids.append(chunk["chunk_id"])

            entities = reference.get("entities", [])
            entity_ids = [e.get("id") for e in entities if isinstance(e, dict) and "id" in e]

            relationships = reference.get("relationships", [])
            rel_ids = [r.get("id") for r in relationships if isinstance(r, dict) and "id" in r]

            if chunk_ids:
                return get_knowledge_graph_for_ids(entity_ids, rel_ids, chunk_ids)

        if isinstance(message, str) and "<think>" in message and "</think>" in message:
            think_pattern = r"<think>.*?</think>"
            message = re.sub(think_pattern, "", message, flags=re.DOTALL).strip()

        entity_ids = []
        rel_ids = []
        chunk_ids = []

        entity_match = re.search(r"['\"]?Entities['\"]?\s*:\s*\[(.*?)\]", message, re.DOTALL)
        if entity_match:
            for part in entity_match.group(1).strip().split(","):
                clean = part.strip().strip("'\"")
                if clean:
                    entity_ids.append(int(clean) if clean.isdigit() else clean)

        rel_match = re.search(r"['\"]?(?:Relationships|Reports)['\"]?\s*:\s*\[(.*?)\]", message, re.DOTALL)
        if rel_match:
            for part in rel_match.group(1).strip().split(","):
                clean = part.strip().strip("'\"")
                if clean:
                    rel_ids.append(int(clean) if clean.isdigit() else clean)

        chunk_match = re.search(r"['\"]?Chunks['\"]?\s*:\s*\[(.*?)\]", message, re.DOTALL)
        if chunk_match:
            chunks_str = chunk_match.group(1).strip()
            if "'" in chunks_str or '"' in chunks_str:
                chunk_ids = re.findall(r"['\"]([^'\"]*)['\"]", chunks_str)
            else:
                chunk_ids = [p.strip() for p in chunks_str.split(",") if p.strip()]

        return get_knowledge_graph_for_ids(entity_ids, rel_ids, chunk_ids)

    except Exception as e:
        print(f"Failed to extract knowledge graph data: {str(e)}")
        traceback.print_exc()
        return {"nodes": [], "links": []}


def check_entity_existence(entity_ids: List[Any]) -> List:
    try:
        query = """
        UNWIND $ids AS id
        OPTIONAL MATCH (e:__Entity__)
        WHERE e.id = id OR e.id = toString(id) OR toString(e.id) = toString(id)
        RETURN id AS input_id, e.id AS found_id, labels(e) AS labels
        """
        result = driver.execute_query(query, {"ids": entity_ids})
        if result.records:
            return [r.get("found_id") for r in result.records if r.get("found_id") is not None]
        return []
    except Exception as e:
        print(f"Error checking entity ID: {str(e)}")
        return []


def get_entities_from_chunk(chunk_id: str) -> List:
    try:
        query = """
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE c.id = $chunk_id
        RETURN collect(distinct e.id) AS entity_ids
        """
        result = driver.execute_query(query, {"chunk_id": chunk_id})
        if result.records and len(result.records) > 0:
            return result.records[0].get("entity_ids", [])
        return []
    except Exception as e:
        print(f"Error querying entities associated with chunk: {str(e)}")
        return []


def get_graph_from_chunks(chunk_ids: List[str]) -> Dict:
    try:
        query = """
        MATCH (c:__Chunk__)-[:MENTIONS]->(e:__Entity__)
        WHERE c.id IN $chunk_ids
        WITH collect(DISTINCT e) AS entities
        UNWIND entities AS e1
        UNWIND entities AS e2
        WITH entities, e1, e2 WHERE e1.id < e2.id
        OPTIONAL MATCH (e1)-[r]-(e2)
        WITH entities, e1, e2, collect(r) AS rels
        WITH entities,
             collect({source: e1.id, target: e2.id, rels: rels}) AS relations
        WITH entities,
             [rel IN relations WHERE size(rel.rels) > 0 |
              [r IN rel.rels | {source: rel.source, target: rel.target, relType: type(r), label: type(r), weight: 1}]
             ] AS links_nested
        WITH entities,
             REDUCE(acc = [], list IN links_nested | acc + list) AS all_links
        WITH entities,
             [link IN all_links | link.source + '_' + link.target + '_' + link.relType] AS link_keys,
             all_links
        WITH entities,
             [i IN RANGE(0, size(all_links)-1) WHERE
              i = REDUCE(min_i = i, j IN RANGE(0, size(all_links)-1) |
                   CASE WHEN link_keys[j] = link_keys[i] AND j < min_i THEN j ELSE min_i END)
             | all_links[i]] AS unique_links
        RETURN
        [e IN entities | {
            id: e.id, label: e.id,
            description: CASE WHEN e.description IS NULL THEN '' ELSE e.description END,
            group: CASE WHEN [lbl IN labels(e) WHERE lbl <> '__Entity__'] <> []
                        THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0] ELSE 'Unknown' END
        }] AS nodes,
        [link IN unique_links | {source: link.source, target: link.target, label: link.label, weight: link.weight}] AS links
        """
        result = driver.execute_query(query, {"chunk_ids": chunk_ids})
        if not result.records or len(result.records) == 0:
            return {"nodes": [], "links": []}
        record = result.records[0]
        return {"nodes": record.get("nodes", []), "links": record.get("links", [])}
    except Exception as e:
        print(f"Failed to get knowledge graph from chunks: {str(e)}")
        return {"nodes": [], "links": []}


def get_knowledge_graph_for_ids(entity_ids=None, relationship_ids=None, chunk_ids=None) -> Dict:
    try:
        entity_ids = entity_ids or []
        relationship_ids = relationship_ids or []
        chunk_ids = chunk_ids or []

        if chunk_ids and not entity_ids:
            for chunk_id in chunk_ids:
                entity_ids.extend(get_entities_from_chunk(chunk_id))
            entity_ids = list(set(entity_ids))

        if not entity_ids and not chunk_ids:
            return {"nodes": [], "links": []}

        verified_entity_ids = check_entity_existence(entity_ids)
        if not verified_entity_ids:
            if chunk_ids:
                return get_graph_from_chunks(chunk_ids)
            return {"nodes": [], "links": []}

        params = {"entity_ids": verified_entity_ids, "max_distance": 1}

        query = """
        MATCH (e:__Entity__)
        WHERE e.id IN $entity_ids
        WITH collect(e) AS base_entities
        UNWIND base_entities AS e1
        UNWIND base_entities AS e2
        WITH base_entities, e1, e2 WHERE e1.id < e2.id
        OPTIONAL MATCH (e1)-[r]-(e2)
        WITH base_entities, e1, e2, collect(r) AS rels
        UNWIND base_entities AS base_entity
        OPTIONAL MATCH (base_entity)-[r1]-(neighbor:__Entity__)
        WHERE NOT neighbor IN base_entities
        WITH base_entities,
             collect(DISTINCT {source: e1.id, target: e2.id, rels: rels}) AS internal_rels,
             collect(DISTINCT neighbor) AS neighbors,
             collect(DISTINCT {source: base_entity.id, target: neighbor.id, rel: r1}) AS external_rels
        WITH base_entities + neighbors AS all_entities, internal_rels, external_rels
        WITH all_entities,
             [rel IN internal_rels WHERE size(rel.rels) > 0 |
              [r IN rel.rels | {source: rel.source, target: rel.target, label: type(r), relType: type(r),
               weight: CASE WHEN r.weight IS NULL THEN 1 ELSE r.weight END}]
             ] AS internal_links_nested,
             [rel IN external_rels WHERE rel.rel IS NOT NULL |
              {source: rel.source, target: rel.target, label: type(rel.rel), relType: type(rel.rel),
               weight: CASE WHEN rel.rel.weight IS NULL THEN 1 ELSE rel.rel.weight END}
             ] AS external_links
        WITH all_entities,
             [link IN external_links | link] +
             [link IN REDUCE(acc = [], list IN internal_links_nested | acc + list) | link] AS all_links_raw
        WITH all_entities,
             [link IN all_links_raw | link.source + '_' + link.target + '_' + link.relType] AS link_keys,
             all_links_raw
        WITH all_entities,
             [i IN RANGE(0, size(all_links_raw)-1) WHERE
              i = REDUCE(min_i = i, j IN RANGE(0, size(all_links_raw)-1) |
                   CASE WHEN link_keys[j] = link_keys[i] AND j < min_i THEN j ELSE min_i END)
             | all_links_raw[i]] AS unique_links
        RETURN
        [n IN all_entities | {
            id: n.id,
            label: CASE WHEN n.id IS NULL THEN "Unknown" ELSE n.id END,
            description: CASE WHEN n.description IS NULL THEN '' ELSE n.description END,
            group: CASE WHEN [lbl IN labels(n) WHERE lbl <> '__Entity__'] <> []
                        THEN [lbl IN labels(n) WHERE lbl <> '__Entity__'][0] ELSE 'Unknown' END
        }] AS nodes,
        [link IN unique_links | {source: link.source, target: link.target, label: link.label, weight: link.weight}] AS links
        """

        result = driver.execute_query(query, params)

        if not result.records or len(result.records) == 0:
            if chunk_ids:
                return get_graph_from_chunks(chunk_ids)
            return {"nodes": [], "links": []}

        record = result.records[0]
        return {"nodes": record.get("nodes", []), "links": record.get("links", [])}

    except Exception as e:
        print(f"Failed to get knowledge graph: {str(e)}")
        if chunk_ids:
            return get_graph_from_chunks(chunk_ids)
        return {"nodes": [], "links": []}


def get_knowledge_graph(limit: int = 100, query: str = None) -> Dict:
    try:
        limit = int(limit) if limit else 100
        params = {"limit": limit}

        if query:
            query_conditions = "WHERE n.id CONTAINS $query OR n.description CONTAINS $query"
            params["query"] = query
        else:
            query_conditions = ""

        node_query = f"""
        MATCH (n:__Entity__)
        {query_conditions}
        WITH n LIMIT $limit
        WITH collect(n) AS entities
        CALL {{
            WITH entities
            MATCH (e1:__Entity__)-[r]-(e2:__Entity__)
            WHERE e1 IN entities AND e2 IN entities AND e1.id < e2.id
            RETURN collect(r) AS relationships
        }}
        RETURN
        [entity IN entities | {{
            id: entity.id, label: entity.id, description: entity.description,
            group: CASE WHEN [lbl IN labels(entity) WHERE lbl <> '__Entity__'] <> []
                        THEN [lbl IN labels(entity) WHERE lbl <> '__Entity__'][0] ELSE 'Unknown' END
        }}] AS nodes,
        [r IN relationships | {{
            source: startNode(r).id, target: endNode(r).id, label: type(r),
            weight: CASE WHEN r.weight IS NOT NULL THEN r.weight ELSE 1 END
        }}] AS links
        """

        result = driver.execute_query(node_query, params)
        if not result or not result.records:
            return {"nodes": [], "links": []}
        record = result.records[0]
        return {"nodes": record["nodes"] or [], "links": record["links"] or []}

    except Exception as e:
        print(f"Failed to get knowledge graph data: {str(e)}")
        return {"error": str(e), "nodes": [], "links": []}


def get_source_content(source_id: str) -> str:
    try:
        if not source_id:
            return "No valid source ID provided"

        if len(source_id) == 40:
            query = "MATCH (n:__Chunk__) WHERE n.id = $id RETURN n.fileName AS fileName, n.text AS text"
            params = {"id": source_id}
        else:
            id_parts = source_id.split(",")
            if len(id_parts) >= 2 and id_parts[0] == "2":
                query = "MATCH (n:__Chunk__) WHERE n.id = $id RETURN n.fileName AS fileName, n.text AS text"
                params = {"id": id_parts[-1]}
            else:
                query = "MATCH (n:__Community__) WHERE n.id = $id RETURN n.summary AS summary, n.full_content AS full_content"
                params = {"id": id_parts[1] if len(id_parts) > 1 else source_id}

        result = driver.execute_query(query, params, result_transformer_=Result.to_df)

        if result is not None and result.shape[0] > 0:
            if "text" in result.columns:
                return f"File name: {result.iloc[0]['fileName']}\n\n{result.iloc[0]['text']}"
            else:
                return f"Summary:\n{result.iloc[0]['summary']}\n\nFull content:\n{result.iloc[0]['full_content']}"
        return f"No relevant content found: source ID {source_id}"

    except Exception as e:
        print(f"Error getting source content: {str(e)}")
        return f"Error occurred while retrieving source content: {str(e)}"


def get_source_file_info(source_id: str) -> dict:
    try:
        if not source_id:
            return {"file_name": "Unknown file"}

        if len(source_id) == 40:
            query = "MATCH (n:__Chunk__) WHERE n.id = $id RETURN n.fileName AS fileName"
            params = {"id": source_id}
        else:
            id_parts = source_id.split(",")
            if len(id_parts) >= 2 and id_parts[0] == "2":
                query = "MATCH (n:__Chunk__) WHERE n.id = $id RETURN n.fileName AS fileName"
                params = {"id": id_parts[-1]}
            else:
                query = 'MATCH (n:__Community__) WHERE n.id = $id RETURN "Community summary" AS fileName'
                params = {"id": id_parts[1] if len(id_parts) > 1 else source_id}

        result = driver.execute_query(query, params, result_transformer_=Result.to_df)

        if result is not None and result.shape[0] > 0 and "fileName" in result.columns:
            file_name = result.iloc[0]["fileName"]
            return {"file_name": os.path.basename(file_name) if file_name else "Unknown file"}
        return {"file_name": f"Source text {source_id}"}

    except Exception as e:
        print(f"Error getting source file info: {str(e)}")
        return {"file_name": f"Source text {source_id}"}


def get_chunks(limit: int = 10, offset: int = 0):
    try:
        query = """
        MATCH (c:__Chunk__)
        RETURN c.id AS id, c.fileName AS fileName, c.text AS text
        ORDER BY c.fileName, c.id
        SKIP $offset
        LIMIT $limit
        """
        result = driver.execute_query(
            query,
            parameters_={"limit": int(limit), "offset": int(offset)},
            result_transformer_=Result.to_df,
        )
        if result is not None and not result.empty:
            return {"chunks": result.to_dict(orient="records"), "total": len(result)}
        return {"chunks": [], "total": 0}
    except Exception as e:
        print(f"Failed to get chunks: {str(e)}")
        return {"error": str(e), "chunks": []}


# ---------------------------------------------------------------------------
# Graph reasoning functions
# ---------------------------------------------------------------------------

def get_shortest_path(driver, entity_a, entity_b, max_hops=3):
    try:
        depth_map = {1: "[*..1]", 2: "[*..2]", 3: "[*..3]", 4: "[*..4]"}
        path_pattern = depth_map.get(max_hops, "[*..5]" if max_hops >= 5 else "[*..3]")

        query = f"""
        MATCH (a:__Entity__), (b:__Entity__)
        WHERE a.id = $entity_a AND b.id = $entity_b
        MATCH p = shortestPath((a)-{path_pattern}-(b))
        RETURN p
        """
        result = driver.execute_query(query, {"entity_a": entity_a, "entity_b": entity_b})

        nodes, links, node_ids = [], [], set()
        path_length = 0
        path_info = f"Shortest path from {entity_a} to {entity_b}"

        if result.records and len(result.records) > 0:
            path = result.records[0].get("p")
            if path:
                for node in path.nodes:
                    node_id = node.get("id")
                    if node_id not in node_ids:
                        node_ids.add(node_id)
                        group = [lbl for lbl in node.labels if lbl != "__Entity__"]
                        nodes.append({"id": node_id, "label": node_id, "description": node.get("description", ""), "group": group[0] if group else "Unknown"})
                for rel in path.relationships:
                    links.append({"source": rel.start_node.get("id"), "target": rel.end_node.get("id"), "label": rel.type, "weight": 1})
                    path_length += 1

        return {"nodes": nodes, "links": links, "path_info": path_info, "path_length": path_length}
    except Exception as e:
        print(f"Failed to get shortest path: {str(e)}")
        return {"nodes": [], "links": [], "error": str(e)}


def get_one_two_hop_paths(driver, entity_a, entity_b):
    try:
        query = """
        MATCH p = (a:__Entity__)-[*1..2]-(b:__Entity__)
        WHERE a.id = $entity_a AND b.id = $entity_b
        RETURN p
        """
        result = driver.execute_query(query, {"entity_a": entity_a, "entity_b": entity_b})

        nodes, links, paths_info = [], [], []
        node_map, link_map = {}, {}

        for record in result.records:
            path = record.get("p")
            if not path:
                continue
            path_desc = []
            for node in path.nodes:
                node_id = node.get("id")
                if node_id not in node_map:
                    group = [lbl for lbl in node.labels if lbl != "__Entity__"]
                    node_data = {"id": node_id, "label": node_id, "description": node.get("description", ""), "group": group[0] if group else "Unknown"}
                    nodes.append(node_data)
                    node_map[node_id] = node_data
            prev_node = None
            for node in path.nodes:
                current_id = node.get("id")
                if prev_node:
                    for rel in path.relationships:
                        start_id = rel.start_node.get("id")
                        end_id = rel.end_node.get("id")
                        if (start_id == prev_node and end_id == current_id) or (start_id == current_id and end_id == prev_node):
                            link_key = f"{start_id}_{end_id}_{rel.type}"
                            if link_key not in link_map:
                                link_data = {"source": start_id, "target": end_id, "label": rel.type, "weight": 1}
                                links.append(link_data)
                                link_map[link_key] = link_data
                            path_desc.append(f"{prev_node} -[{rel.type}]-> {current_id}")
                prev_node = current_id
            if path_desc:
                path_str = " ".join(path_desc)
                if path_str not in paths_info:
                    paths_info.append(path_str)

        return {"nodes": nodes, "links": links, "paths_info": paths_info, "path_count": len(paths_info)}
    except Exception as e:
        print(f"Failed to get 1-to-2-hop paths: {str(e)}")
        return {"nodes": [], "links": [], "error": str(e)}


def get_common_neighbors(driver, entity_a, entity_b):
    try:
        query = """
        MATCH (a:__Entity__ {id: $entity_a})--(x)--(b:__Entity__ {id: $entity_b})
        RETURN DISTINCT x
        """
        result = driver.execute_query(query, {"entity_a": entity_a, "entity_b": entity_b})

        nodes = [
            {"id": entity_a, "label": entity_a, "group": "Source", "description": ""},
            {"id": entity_b, "label": entity_b, "group": "Target", "description": ""},
        ]
        links = []
        common_neighbors = []
        node_ids = {entity_a, entity_b}

        for record in result.records:
            neighbor = record.get("x")
            if not neighbor:
                continue
            neighbor_id = neighbor.get("id")
            common_neighbors.append(neighbor_id)
            if neighbor_id not in node_ids:
                node_ids.add(neighbor_id)
                group = [lbl for lbl in neighbor.labels if lbl != "__Entity__"]
                nodes.append({"id": neighbor_id, "label": neighbor_id, "description": neighbor.get("description", ""), "group": group[0] if group else "Common"})
            links.append({"source": entity_a, "target": neighbor_id, "label": "connected", "weight": 1})
            links.append({"source": neighbor_id, "target": entity_b, "label": "connected", "weight": 1})

        return {"nodes": nodes, "links": links, "common_neighbors": common_neighbors, "neighbor_count": len(common_neighbors)}
    except Exception as e:
        print(f"Failed to get common neighbors: {str(e)}")
        return {"nodes": [], "links": [], "error": str(e)}


def get_all_paths(driver, entity_a, entity_b, max_depth=3):
    try:
        check_result = driver.execute_query(
            "MATCH (a:__Entity__), (b:__Entity__) WHERE a.id = $entity_a AND b.id = $entity_b RETURN a.id AS id_a, b.id AS id_b",
            {"entity_a": entity_a, "entity_b": entity_b},
        )
        if not check_result.records:
            return {"error": f"Entity '{entity_a}' or '{entity_b}' does not exist", "nodes": [], "links": []}

        depth_map = {1: "[*1..1]", 2: "[*1..2]", 3: "[*1..3]", 4: "[*1..4]"}
        path_pattern = depth_map.get(max_depth, "[*1..5]" if max_depth >= 5 else "[*1..3]")

        query = f"""
        MATCH p = (a:__Entity__)-{path_pattern}-(b:__Entity__)
        WHERE a.id = $entity_a AND b.id = $entity_b
        RETURN p LIMIT 10
        """
        result = driver.execute_query(query, {"entity_a": entity_a, "entity_b": entity_b})

        nodes, links, paths_info = [], [], []
        node_map, link_map = {}, {}

        for record in result.records:
            path = record.get("p")
            if not path:
                continue
            for node in path.nodes:
                node_id = node.get("id")
                if node_id not in node_map:
                    group = [lbl for lbl in node.labels if lbl != "__Entity__"]
                    node_data = {"id": node_id, "label": node_id, "description": node.get("description", ""), "group": group[0] if group else "Unknown"}
                    nodes.append(node_data)
                    node_map[node_id] = node_data
            path_rels = []
            for rel in path.relationships:
                start_id = rel.start_node.get("id")
                end_id = rel.end_node.get("id")
                link_key = f"{start_id}_{end_id}_{rel.type}"
                if link_key not in link_map:
                    link_data = {"source": start_id, "target": end_id, "label": rel.type, "weight": 1}
                    links.append(link_data)
                    link_map[link_key] = link_data
                path_rels.append((start_id, rel.type, end_id))
            if path_rels:
                path_str = " -> ".join([f"{s} -[{r}]-> {e}" for s, r, e in path_rels])
                if path_str not in paths_info:
                    paths_info.append(path_str)

        return {"nodes": nodes, "links": links, "paths_info": paths_info, "path_count": len(paths_info)}
    except Exception as e:
        print(f"Failed to get all paths: {str(e)}")
        return {"nodes": [], "links": [], "error": str(e)}


def get_entity_cycles(driver, entity_id, max_depth=4):
    try:
        depth_map = {1: "[*1..1]", 2: "[*1..2]", 3: "[*1..3]", 4: "[*1..4]"}
        path_pattern = depth_map.get(max_depth, "[*1..4]")

        query = f"""
        MATCH p = (a:__Entity__)-{path_pattern}->(a)
        WHERE a.id = $entity_id
        RETURN p LIMIT 10
        """
        result = driver.execute_query(query, {"entity_id": entity_id})

        nodes, links, cycles_info = [], [], []
        node_map, link_map = {}, {}

        for record in result.records:
            path = record.get("p")
            if not path:
                continue
            for node in path.nodes:
                node_id = node.get("id")
                if node_id not in node_map:
                    group = [lbl for lbl in node.labels if lbl != "__Entity__"]
                    node_data = {"id": node_id, "label": node_id, "description": node.get("description", ""), "group": group[0] if group else "Unknown"}
                    nodes.append(node_data)
                    node_map[node_id] = node_data
            cycle_rels = []
            for rel in path.relationships:
                start_id = rel.start_node.get("id")
                end_id = rel.end_node.get("id")
                link_key = f"{start_id}_{end_id}_{rel.type}"
                if link_key not in link_map:
                    link_data = {"source": start_id, "target": end_id, "label": rel.type, "weight": 1}
                    links.append(link_data)
                    link_map[link_key] = link_data
                cycle_rels.append((start_id, rel.type, end_id))
            if cycle_rels:
                cycle_str = " -> ".join([f"{s} -[{r}]-> {e}" for s, r, e in cycle_rels])
                if cycle_str not in [c["description"] for c in cycles_info]:
                    cycles_info.append({"description": cycle_str, "length": len(cycle_rels)})

        return {"nodes": nodes, "links": links, "cycles_info": cycles_info, "cycle_count": len(cycles_info)}
    except Exception as e:
        print(f"Failed to find cycles: {str(e)}")
        return {"nodes": [], "links": [], "error": str(e)}


def get_entity_influence(driver, entity_id, max_depth=2):
    try:
        depth_map = {1: "[*1..1]", 2: "[*1..2]", 3: "[*1..3]"}
        path_pattern = depth_map.get(max_depth, "[*1..2]")

        query = f"""
        MATCH p = (a:__Entity__)-{path_pattern}-(b:__Entity__)
        WHERE a.id = $entity_id
        RETURN p LIMIT 100
        """
        result = driver.execute_query(query, {"entity_id": entity_id})

        center_node = {"id": entity_id, "label": entity_id, "description": "", "group": "Center"}
        nodes = [center_node]
        links = []
        node_map = {entity_id: center_node}
        link_map = {}
        direct_connections = set()
        connection_types = {}

        for record in result.records:
            path = record.get("p")
            if not path:
                continue
            for i, node in enumerate(path.nodes):
                node_id = node.get("id")
                if node_id != entity_id and node_id not in node_map:
                    group = [lbl for lbl in node.labels if lbl != "__Entity__"]
                    if i == 1 or i == len(path.nodes) - 2:
                        group_val = "Level1"
                        direct_connections.add(node_id)
                    else:
                        group_val = f"Level{min(i, len(path.nodes) - i - 1)}"
                    node_data = {"id": node_id, "label": node_id, "description": node.get("description", ""), "group": group_val}
                    nodes.append(node_data)
                    node_map[node_id] = node_data
            for rel in path.relationships:
                start_id = rel.start_node.get("id")
                end_id = rel.end_node.get("id")
                rel_type = rel.type
                connection_types[rel_type] = connection_types.get(rel_type, 0) + 1
                link_key = f"{start_id}_{end_id}_{rel_type}"
                if link_key not in link_map:
                    link_data = {"source": start_id, "target": end_id, "label": rel_type, "weight": 1}
                    links.append(link_data)
                    link_map[link_key] = link_data

        influence_stats = {
            "direct_connections": len(direct_connections),
            "total_connections": len(nodes) - 1,
            "connection_types": [{"type": k, "count": v} for k, v in connection_types.items()],
            "relation_distribution": connection_types,
        }
        return {"nodes": nodes, "links": links, "influence_stats": influence_stats}
    except Exception as e:
        print(f"Failed to analyze entity influence range: {str(e)}")
        return {"nodes": [], "links": [], "error": str(e)}


def get_simplified_community(driver, entity_id, max_depth=2):
    try:
        check_result = driver.execute_query(
            "MATCH (a:__Entity__) WHERE a.id = $entity_id RETURN a.id AS id, labels(a) AS labels",
            {"entity_id": entity_id},
        )
        if not check_result.records:
            return {"error": f"Entity '{entity_id}' does not exist", "nodes": [], "links": []}

        depth_map = {1: "[*0..1]", 2: "[*0..2]", 3: "[*0..3]"}
        path_pattern = depth_map.get(max_depth, "[*0..2]")

        neighbors_result = driver.execute_query(
            f"MATCH p = (a:__Entity__)-{path_pattern}-(b:__Entity__) WHERE a.id = $entity_id RETURN DISTINCT b LIMIT 100",
            {"entity_id": entity_id},
        )

        nodes = []
        node_map = {}
        entity_ids = []

        for record in neighbors_result.records:
            entity = record.get("b")
            if not entity:
                continue
            try:
                if hasattr(entity, "get"):
                    node_id = entity.get("id")
                    node_labels = entity.get("labels", entity.get("_labels", []))
                elif hasattr(entity, "id"):
                    node_id = entity.id
                    node_labels = getattr(entity, "labels", [])
                else:
                    node_id = str(entity)
                    node_labels = []

                if node_id and node_id not in node_map:
                    group = "Unknown"
                    if isinstance(node_labels, (list, set, frozenset)):
                        non_entity = [lbl for lbl in node_labels if lbl != "__Entity__"]
                        if non_entity:
                            group = non_entity[0]
                    if node_id == entity_id:
                        group = "Center"
                    node_data = {"id": node_id, "label": node_id, "description": "", "group": group}
                    nodes.append(node_data)
                    node_map[node_id] = node_data
                    entity_ids.append(node_id)
            except Exception:
                continue

        # Get relationships between found entities
        links = []
        if len(entity_ids) >= 2:
            try:
                rel_result = driver.execute_query(
                    """
                    MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
                    WHERE e1.id IN $ids AND e2.id IN $ids
                    RETURN e1.id AS source, type(r) AS rel_type, e2.id AS target
                    LIMIT 200
                    """,
                    {"ids": entity_ids},
                )
                for record in rel_result.records:
                    links.append({
                        "source": record.get("source"),
                        "target": record.get("target"),
                        "label": record.get("rel_type"),
                        "weight": 1,
                    })
            except Exception as e:
                print(f"Failed to get community relationships: {str(e)}")

        stats = {
            "id": entity_id,
            "entity_count": len(nodes),
            "relation_count": len(links),
            "summary": f"{max_depth}-hop neighbor community of entity {entity_id}",
        }
        return {"nodes": nodes, "links": links, "community_info": stats}

    except Exception as e:
        print(f"Failed to get simplified community: {str(e)}")
        traceback.print_exc()
        return {"nodes": [], "links": [], "error": str(e)}
