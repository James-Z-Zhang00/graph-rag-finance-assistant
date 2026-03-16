"""
Knowledge-graph router — direct Neo4j access.
Ported from the monolith server/routers/knowledge_graph.py.
"""

import traceback
from typing import Optional
from fastapi import APIRouter, HTTPException

from db.connection import get_db_manager
from services.kg_service import (
    get_knowledge_graph,
    extract_kg_from_message,
    get_chunks,
    get_shortest_path,
    get_one_two_hop_paths,
    get_common_neighbors,
    get_all_paths,
    get_entity_cycles,
    get_entity_influence,
    get_simplified_community,
)
from models.schemas import (
    ReasoningRequest,
    EntityData, EntityDeleteData, EntitySearchFilter, EntityUpdateData,
    RelationData, RelationDeleteData, RelationSearchFilter, RelationUpdateData,
)

router = APIRouter()


@router.get("/knowledge_graph")
async def knowledge_graph(limit: int = 100, query: Optional[str] = None):
    return get_knowledge_graph(limit, query)


@router.get("/knowledge_graph_from_message")
async def knowledge_graph_from_message(message: Optional[str] = None, query: Optional[str] = None):
    if not message:
        return {"nodes": [], "links": []}
    return extract_kg_from_message(message, query)


@router.get("/chunks")
async def chunks(limit: int = 10, offset: int = 0):
    return get_chunks(limit, offset)


@router.post("/kg_reasoning")
async def knowledge_graph_reasoning(request: ReasoningRequest):
    try:
        db_manager = get_db_manager()
        driver = db_manager.get_driver()

        reasoning_type = request.reasoning_type
        entity_a = request.entity_a.strip()
        entity_b = request.entity_b.strip() if request.entity_b else None
        max_depth = min(max(1, request.max_depth), 5)
        algorithm = request.algorithm

        if reasoning_type == "entity_community":
            return await _process_community_detection(entity_a, max_depth, algorithm)

        if reasoning_type == "shortest_path":
            if not entity_b:
                return {"error": "Shortest path query requires two entities", "nodes": [], "links": []}
            result = get_shortest_path(driver, entity_a, entity_b, max_depth)
        elif reasoning_type == "one_two_hop":
            if not entity_b:
                return {"error": "1-to-2-hop query requires two entities", "nodes": [], "links": []}
            result = get_one_two_hop_paths(driver, entity_a, entity_b)
        elif reasoning_type == "common_neighbors":
            if not entity_b:
                return {"error": "Common neighbors query requires two entities", "nodes": [], "links": []}
            result = get_common_neighbors(driver, entity_a, entity_b)
        elif reasoning_type == "all_paths":
            if not entity_b:
                return {"error": "All paths query requires two entities", "nodes": [], "links": []}
            result = get_all_paths(driver, entity_a, entity_b, max_depth)
        elif reasoning_type == "entity_cycles":
            result = get_entity_cycles(driver, entity_a, max_depth)
        elif reasoning_type == "entity_influence":
            result = get_entity_influence(driver, entity_a, max_depth)
        else:
            return {"error": "Unknown reasoning type", "nodes": [], "links": []}

        return result
    except Exception as e:
        print(f"Reasoning query exception: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "nodes": [], "links": []}


async def _process_community_detection(entity_id: str, max_depth: int, algorithm: str):
    try:
        community_info = await _get_entity_community_from_db(entity_id)
        if community_info and community_info.get("nodes") and community_info.get("links"):
            return community_info
        db_manager = get_db_manager()
        driver = db_manager.get_driver()
        return get_simplified_community(driver, entity_id, max_depth)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "nodes": [], "links": []}


async def _get_entity_community_from_db(entity_id: str):
    try:
        db_manager = get_db_manager()
        driver = db_manager.get_driver()

        community_result = driver.execute_query(
            "MATCH (e:__Entity__ {id: $entity_id})-[:IN_COMMUNITY]->(c:__Community__) RETURN c.id AS community_id LIMIT 1",
            {"entity_id": entity_id},
        )
        if not community_result.records:
            return None

        community_id = community_result.records[0].get("community_id")
        if not community_id:
            return None

        community_data = driver.execute_query(
            """
            MATCH (c:__Community__ {id: $community_id})<-[:IN_COMMUNITY]-(e:__Entity__)
            WITH c, collect({id: e.id, description: e.description, labels: labels(e)}) AS entities
            OPTIONAL MATCH (c) WHERE c.summary IS NOT NULL
            CALL {
                WITH c
                MATCH (c)<-[:IN_COMMUNITY]-(e1:__Entity__)-[r]->(e2:__Entity__)-[:IN_COMMUNITY]->(c)
                RETURN collect({source: e1.id, target: e2.id, type: type(r)}) AS relationships
            }
            RETURN c.id AS community_id, c.summary AS summary, entities, relationships
            """,
            {"community_id": community_id},
        )
        if not community_data.records:
            return None

        row = community_data.records[0]
        summary = row.get("summary", "No community summary")
        nodes, links = [], []

        for entity in row.get("entities", []):
            entity_labels = entity.get("labels", [])
            group = [lbl for lbl in entity_labels if lbl != "__Entity__"]
            group = group[0] if group else "Unknown"
            if entity.get("id") == entity_id:
                group = "Center"
            nodes.append({"id": entity.get("id"), "label": entity.get("id"), "description": entity.get("description", ""), "group": group})

        for rel in row.get("relationships", []):
            links.append({"source": rel.get("source"), "target": rel.get("target"), "label": rel.get("type"), "weight": 1})

        stats = {"id": community_id, "entity_count": len(nodes), "relation_count": len(links), "summary": summary}
        return {"nodes": nodes, "links": links, "community_info": stats}

    except Exception as e:
        print(f"Failed to get community info: {str(e)}")
        return None


@router.get("/entity_types")
def get_entity_types():
    db_manager = get_db_manager()
    try:
        result = db_manager.execute_query("""
        MATCH (e:__Entity__)
        RETURN DISTINCT
        CASE WHEN size(labels(e)) > 1
             THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0]
             ELSE 'Unknown'
        END AS entity_type
        ORDER BY entity_type
        """)
        entity_types = result["entity_type"].tolist() if "entity_type" in result.columns else []
        return {"entity_types": entity_types}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get entity types: {str(e)}")


@router.get("/relation_types")
def get_relation_types():
    db_manager = get_db_manager()
    try:
        result = db_manager.execute_query("""
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) AS relation_type
        ORDER BY relation_type
        """)
        relation_types = result["relation_type"].tolist() if "relation_type" in result.columns else []
        return {"relation_types": relation_types}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get relation types: {str(e)}")


@router.post("/entities/search")
def search_entities(filters: EntitySearchFilter):
    db_manager = get_db_manager()
    try:
        conditions = ["e:__Entity__"]
        params = {}
        if filters.type:
            conditions.append(f"e:{filters.type}")
        if filters.term:
            conditions.append("e.id CONTAINS $term")
            params["term"] = filters.term

        query = f"""
        MATCH (e)
        WHERE {' AND '.join(conditions)}
        RETURN e.id AS id,
               COALESCE(e.id, '') AS name,
               CASE WHEN size(labels(e)) > 1
                    THEN [lbl IN labels(e) WHERE lbl <> '__Entity__'][0]
                    ELSE 'Unknown'
               END AS type,
               COALESCE(e.description, '') AS description
        LIMIT {filters.limit}
        """
        result = db_manager.execute_query(query, params)

        if result is None:
            return {"entities": []}

        entities = []
        if not result.empty:
            for _, row in result.iterrows():
                entities.append({
                    "id": row.get("id", ""),
                    "name": row.get("name", ""),
                    "type": row.get("type", "Unknown"),
                    "description": row.get("description", ""),
                    "properties": {},
                })
        return {"entities": entities}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to search entities: {str(e)}")


@router.post("/relations/search")
def search_relations(filters: RelationSearchFilter):
    db_manager = get_db_manager()
    try:
        conditions = []
        params = {}
        if filters.source:
            conditions.append("e1.id = $source")
            params["source"] = filters.source
        if filters.target:
            conditions.append("e2.id = $target")
            params["target"] = filters.target
        if filters.type:
            conditions.append("type(r) = $relType")
            params["relType"] = filters.type

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"""
        MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
        {where_clause}
        RETURN e1.id AS source,
               type(r) AS type,
               e2.id AS target,
               COALESCE(r.description, '') AS description,
               COALESCE(r.weight, 0.5) AS weight
        LIMIT {filters.limit}
        """
        result = db_manager.execute_query(query, params)
        relations = []
        if not result.empty:
            for _, row in result.iterrows():
                relations.append({
                    "source": row.get("source"),
                    "type": row.get("type"),
                    "target": row.get("target"),
                    "description": row.get("description", ""),
                    "weight": row.get("weight", 0.5),
                    "properties": {},
                })
        return {"relations": relations}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to search relations: {str(e)}")


@router.post("/entity/create")
def create_entity(entity_data: EntityData):
    db_manager = get_db_manager()
    try:
        check = db_manager.execute_query("MATCH (e:__Entity__ {id: $id}) RETURN count(e) AS count", {"id": entity_data.id})
        if not check.empty and check.iloc[0]["count"] > 0:
            return {"success": False, "message": f"Entity ID '{entity_data.id}' already exists"}

        result = db_manager.execute_query(
            f"CREATE (e:__Entity__:{entity_data.type} {{id: $id, name: $name, description: $description}}) RETURN e.id AS id",
            {"id": entity_data.id, "name": entity_data.name, "description": entity_data.description},
        )
        if not result.empty:
            return {"success": True, "id": result.iloc[0]["id"]}
        return {"success": False, "message": "Failed to create entity"}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Failed to create entity: {str(e)}"}


@router.post("/entity/update")
def update_entity(entity_data: EntityUpdateData):
    db_manager = get_db_manager()
    try:
        check = db_manager.execute_query("MATCH (e:__Entity__ {id: $id}) RETURN count(e) AS count", {"id": entity_data.id})
        if check.empty or check.iloc[0]["count"] == 0:
            return {"success": False, "message": f"Entity ID '{entity_data.id}' does not exist"}

        if entity_data.type is not None:
            labels_result = db_manager.execute_query("MATCH (e:__Entity__ {id: $id}) RETURN labels(e) AS labels", {"id": entity_data.id})
            if not labels_result.empty:
                current_labels = labels_result.iloc[0]["labels"]
                remove_labels = [lbl for lbl in current_labels if lbl != "__Entity__"]
                update_type_query = f"""
                MATCH (e:__Entity__ {{id: $id}})
                {' '.join(f'REMOVE e:{lbl}' for lbl in remove_labels)}
                SET e:{entity_data.type}
                RETURN e.id as id
                """
                db_manager.execute_query(update_type_query, {"id": entity_data.id})

        params = {"id": entity_data.id}
        set_clauses = []
        if entity_data.name is not None:
            set_clauses.append("e.name = $name")
            params["name"] = entity_data.name
        if entity_data.description is not None:
            set_clauses.append("e.description = $description")
            params["description"] = entity_data.description
        if set_clauses:
            db_manager.execute_query(f"MATCH (e:__Entity__ {{id: $id}}) SET {', '.join(set_clauses)} RETURN e.id as id", params)

        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Failed to update entity: {str(e)}"}


@router.post("/entity/delete")
def delete_entity(entity_data: EntityDeleteData):
    db_manager = get_db_manager()
    try:
        check = db_manager.execute_query("MATCH (e:__Entity__ {id: $id}) RETURN count(e) AS count", {"id": entity_data.id})
        if check is None or check.empty:
            return {"success": False, "message": f"实体ID '{entity_data.id}' 不存在"}
        count_value = check.iloc[0]["count"] if "count" in check.columns else 0
        if count_value == 0:
            return {"success": False, "message": f"Entity ID '{entity_data.id}' does not exist"}

        db_manager.execute_query("MATCH (e:__Entity__ {id: $id})-[r]-() DELETE r", {"id": entity_data.id})
        db_manager.execute_query("MATCH (e:__Entity__ {id: $id}) DELETE e", {"id": entity_data.id})
        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Failed to delete entity: {str(e)}"}


@router.post("/relation/create")
def create_relation(relation_data: RelationData):
    db_manager = get_db_manager()
    try:
        check = db_manager.execute_query(
            "MATCH (e1:__Entity__ {id: $source}) MATCH (e2:__Entity__ {id: $target}) RETURN count(e1) AS source_count, count(e2) AS target_count",
            {"source": relation_data.source, "target": relation_data.target},
        )
        if check.empty:
            return {"success": False, "message": "Cannot verify entity existence"}
        if check.iloc[0]["source_count"] == 0:
            return {"success": False, "message": f"Source entity '{relation_data.source}' does not exist"}
        if check.iloc[0]["target_count"] == 0:
            return {"success": False, "message": f"Target entity '{relation_data.target}' does not exist"}

        rel_check = db_manager.execute_query(
            "MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target}) WHERE type(r) = $relType RETURN count(r) AS rel_count",
            {"source": relation_data.source, "target": relation_data.target, "relType": relation_data.type},
        )
        if not rel_check.empty and rel_check.iloc[0]["rel_count"] > 0:
            return {"success": False, "message": "Relation already exists"}

        result = db_manager.execute_query(
            f"""
            MATCH (e1:__Entity__ {{id: $source}})
            MATCH (e2:__Entity__ {{id: $target}})
            CREATE (e1)-[r:{relation_data.type} {{description: $description, weight: $weight}}]->(e2)
            RETURN type(r) AS type
            """,
            {"source": relation_data.source, "target": relation_data.target, "description": relation_data.description, "weight": relation_data.weight},
        )
        if not result.empty:
            return {"success": True, "type": result.iloc[0]["type"]}
        return {"success": False, "message": "Failed to create relation"}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Failed to create relation: {str(e)}"}


@router.post("/relation/update")
def update_relation(relation_data: RelationUpdateData):
    db_manager = get_db_manager()
    try:
        check = db_manager.execute_query(
            "MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target}) WHERE type(r) = $relType RETURN count(r) AS count",
            {"source": relation_data.source, "target": relation_data.target, "relType": relation_data.original_type},
        )
        if check.empty or check.iloc[0]["count"] == 0:
            return {"success": False, "message": "Relation does not exist"}

        if relation_data.new_type and relation_data.new_type != relation_data.original_type:
            props_result = db_manager.execute_query(
                "MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target}) WHERE type(r) = $relType RETURN r.description AS description, r.weight AS weight",
                {"source": relation_data.source, "target": relation_data.target, "relType": relation_data.original_type},
            )
            db_manager.execute_query(
                "MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target}) WHERE type(r) = $relType DELETE r",
                {"source": relation_data.source, "target": relation_data.target, "relType": relation_data.original_type},
            )
            if not props_result.empty:
                props = props_result.iloc[0]
                description = relation_data.description if relation_data.description is not None else props.get("description", "")
                weight = relation_data.weight if relation_data.weight is not None else props.get("weight", 0.5)
            else:
                description = relation_data.description or ""
                weight = relation_data.weight or 0.5
            db_manager.execute_query(
                f"""
                MATCH (e1:__Entity__ {{id: $source}})
                MATCH (e2:__Entity__ {{id: $target}})
                CREATE (e1)-[r:{relation_data.new_type} {{description: $description, weight: $weight}}]->(e2)
                RETURN type(r) AS type
                """,
                {"source": relation_data.source, "target": relation_data.target, "description": description, "weight": weight},
            )
        else:
            params = {"source": relation_data.source, "target": relation_data.target, "relType": relation_data.original_type}
            set_clauses = []
            if relation_data.description is not None:
                set_clauses.append("r.description = $description")
                params["description"] = relation_data.description
            if relation_data.weight is not None:
                set_clauses.append("r.weight = $weight")
                params["weight"] = relation_data.weight
            if set_clauses:
                db_manager.execute_query(
                    f"MATCH (e1:__Entity__ {{id: $source}})-[r]->(e2:__Entity__ {{id: $target}}) WHERE type(r) = $relType SET {', '.join(set_clauses)} RETURN type(r) as type",
                    params,
                )
        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Failed to update relation: {str(e)}"}


@router.post("/relation/delete")
def delete_relation(relation_data: RelationDeleteData):
    db_manager = get_db_manager()
    try:
        check = db_manager.execute_query(
            "MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target}) WHERE type(r) = $relType RETURN count(r) AS count",
            {"source": relation_data.source, "target": relation_data.target, "relType": relation_data.type},
        )
        if check.empty or check.iloc[0]["count"] == 0:
            return {"success": False, "message": "Relation does not exist"}
        db_manager.execute_query(
            "MATCH (e1:__Entity__ {id: $source})-[r]->(e2:__Entity__ {id: $target}) WHERE type(r) = $relType DELETE r",
            {"source": relation_data.source, "target": relation_data.target, "relType": relation_data.type},
        )
        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Failed to delete relation: {str(e)}"}
