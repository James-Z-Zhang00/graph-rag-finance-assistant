"""
Graph construction and community summary prompt templates.

These templates are used in the graph index build and maintenance pipeline.
"""

system_template_build_graph = """
-Goal-
Given a relevant text document and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
-entity_name: Name of the entity, capitalized
-entity_type: One of the following types: [{entity_types}]
-entity_description: A comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
2. From the entities identified in step 1, identify all pairs of entities (source_entity, target_entity) that are *clearly related* to each other.
For each related pair, extract the following information:
-source_entity: Name of the source entity, as identified in step 1
-target_entity: Name of the target entity, as identified in step 1
-relationship_type: One of the following types: [{relationship_types}]. If it cannot be classified under any of the listed types, use the last catch-all type.
-relationship_description: Explanation of why you believe the source entity and target entity are related
-relationship_strength: A numeric score representing the strength of the relationship between the source and target entities
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
3. Output all entity and relationship attributes in English. Output all entities and relationships identified in steps 1 and 2 as a single list, using **{record_delimiter}** as the list delimiter.
4. When complete, output {completion_delimiter}

###################### 
- Examples - 
###################### 
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"study"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:
Text:
When humanity made first contact, it was with a message that couldn't be decoded by any existing system. The resonance was uncanny, a loop that seemed to shift its own parameters, adapting to every attempt at interpretation. The message was alive. The team of physicists and linguists gathered, watching as the patterns continued to rewrite themselves, piece by piece.

The first breakthrough came when Dr. Elena Park noticed a repeating sequence, one that mimicked the phonetic structures of ancient languages... but never quite settled into a recognizable form. Then Sam Rivera, an ethnomusicologist, realized it wasn't a static message at all; it was a dialogue. The signal wasn't just repeating—it was responding.

Every time someone spoke aloud in the chamber, the frequencies shifted, like echoes forming new sentences across a medium that shouldn't possess agency. The room began to feel less like a lab and more like a cathedral.

"It's learning us," Sam whispered, voice trembling. "It's learning how we speak."

Alex, the mission lead, didn't respond. He was already staring at the monitors, watching the patterns unfold. Different voices yielded different responses; emotional inflection seemed to alter the semantic density of the signal. This wasn't just a translation problem. It was an emergent language interface.

And there, in the logs, the beginning of something the team wasn't prepared for: structure. Words, or something like them, building themselves from the raw weave of interference.

It wasn't a message.

It was a bridge.
################
Output:
("entity"{tuple_delimiter}"Dr. Elena Park"{tuple_delimiter}"person"{tuple_delimiter}"Dr. Elena Park is part of a team deciphering a living message from an unknown intelligence, specifically identifying linguistic patterns in the signal."){record_delimiter}
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"contact"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"leads"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"leads"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"controled by"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
Example 3:
Entity_types: [Company, Executive, Filing, FinancialMetric, BusinessSegment, TimePeriod]
Text:
On March 5, 2025, Sarah Chen, age 52, was appointed Executive Vice President and Chief Financial Officer of RetailCo Inc., effective April 1, 2025. Ms. Chen will succeed Robert Mills in this role. The Compensation Committee approved the following changes: Ms. Chen's annualized base salary will be $850,000. For fiscal year 2026, her target annual cash incentive opportunity will be 150% of base salary, with a maximum payout of 200%. She will receive an annual equity award valued at approximately $6,500,000, consisting of 70% performance-based restricted stock units and 30% restricted stock.
################
Output:
("entity"{tuple_delimiter}"Sarah Chen"{tuple_delimiter}"Executive"{tuple_delimiter}"Sarah Chen, age 52, was appointed Executive Vice President and Chief Financial Officer of RetailCo Inc. effective April 1, 2025. Her annualized base salary is $850,000, with a target annual cash incentive of 150% of base salary (maximum 200%) and an annual equity award of approximately $6,500,000 (70% performance-based RSUs, 30% restricted stock)."){record_delimiter}
("entity"{tuple_delimiter}"Robert Mills"{tuple_delimiter}"Executive"{tuple_delimiter}"Robert Mills previously served as Chief Financial Officer of RetailCo Inc. and was succeeded by Sarah Chen effective April 1, 2025."){record_delimiter}
("entity"{tuple_delimiter}"RetailCo Inc."{tuple_delimiter}"Company"{tuple_delimiter}"RetailCo Inc. is a retail corporation that filed a Form 8-K reporting executive leadership changes and compensation arrangements."){record_delimiter}
("entity"{tuple_delimiter}"Fiscal Year 2026"{tuple_delimiter}"TimePeriod"{tuple_delimiter}"Fiscal year 2026, the performance period for Sarah Chen's incentive compensation plan."){record_delimiter}
("relationship"{tuple_delimiter}"Sarah Chen"{tuple_delimiter}"RetailCo Inc."{tuple_delimiter}"EMPLOYED_BY"{tuple_delimiter}"Sarah Chen was appointed CFO of RetailCo Inc. effective April 1, 2025."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Sarah Chen"{tuple_delimiter}"Robert Mills"{tuple_delimiter}"EMPLOYED_BY"{tuple_delimiter}"Sarah Chen succeeded Robert Mills as CFO of RetailCo Inc."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Sarah Chen"{tuple_delimiter}"Fiscal Year 2026"{tuple_delimiter}"PERIOD_COVERS"{tuple_delimiter}"Sarah Chen's incentive compensation targets apply for fiscal year 2026."{tuple_delimiter}7){completion_delimiter}
#############################
"""

human_template_build_graph = """
-Real Data-
######################
Entity types: {entity_types}
Relationship types: {relationship_types}
Text: {input_text}
######################
Output:
"""

system_template_build_index = """
You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which ones should be merged.
These entities may differ slightly in format or content, but essentially refer to the same entity. Use your analytical skills to identify duplicates.
Rules for identifying duplicate entities:
1. Entities with minor semantic differences should be considered duplicates.
2. Entities that differ in format but have the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. Do not merge entities if they refer to different numbers, dates, or product models.
Output format:
1. Output the entities to be merged as a Python list, preserving their original text exactly as given.
2. If there are multiple groups of mergeable entities, output each group as a separate list on its own line.
3. If there are no entities to merge, output an empty list.
4. Output only the list(s); no additional explanation is needed.
5. Do not output nested lists; only output flat lists.
###################### 
- Example - 
###################### 
Example 1:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R', 'Star Ocean: A Research Journey']
#############
Output:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R']
#############################
Example 2:
['Sony', 'Sony Inc', 'Google', 'Google Inc', 'OpenAI']
#############
Output:
['Sony', 'Sony Inc']
['Google', 'Google Inc']
#############################
Example 3:
['December 16, 2023', 'December 2, 2023', 'December 23, 2023', 'December 26, 2023']
Output:
[]
#############################
"""

user_template_build_index = """
The following is the list of entities to process:
{entities}
Please identify duplicate entities and provide a list of entities that can be merged.
Output:
"""

community_template = """
Based on the provided nodes and relationships belonging to the same graph community,
generate a natural language summary of the graph community information:
{community_info}
Summary:
"""

COMMUNITY_SUMMARY_PROMPT = """
Given an input triplet, generate an informative summary. No preamble.
"""

entity_alignment_prompt = """
Given these entities that should refer to the same concept:
{entity_desc}

Which entity ID best represents the canonical form? Reply with only the entity ID."""

__all__ = [
    "system_template_build_graph",
    "human_template_build_graph",
    "system_template_build_index",
    "user_template_build_index",
    "community_template",
    "COMMUNITY_SUMMARY_PROMPT",
    "entity_alignment_prompt",
]
