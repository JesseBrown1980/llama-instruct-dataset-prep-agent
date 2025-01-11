# pipeline.py

import os
import json
import logging
from agents import (
    questioner_agent_invoke,
    question_checker_agent_invoke,
    maker_agent,
    formatter_agent,
    checker_agent
)
from downloader import download_and_clean_resource
from knowledge_graph import build_knowledge_graph

logger = logging.getLogger(__name__)

def process_owl_file_full(
    owl_path,
    limit=5,
    output_kg="knowledge_graph.json",
    output_ds="dolly_dataset.json"
):
    """
    Full pipeline:
      1) Build local knowledge graph from owl_path
      2) Download references
      3) Save knowledge graph to JSON
      4) For each (class or property) up to limit:
         - Combine label+definition as knowledge context
         - Use questioner_agent + question_checker_agent to generate final Q list
         - For each Q: maker_agent -> formatter_agent -> checker_agent -> store
      5) Save final Dolly-format Q&A dataset
    """
    logger.info(f"Starting full OWL processing pipeline for {owl_path}")
    logger.info(f"Parameters: limit={limit}, output_kg={output_kg}, output_ds={output_ds}")
    
    # 1) Build knowledge graph
    kg = build_knowledge_graph(owl_path)

    # 2) Download references
    clean_refs = []
    for ref in kg["refs"]:
        ref_url = str(ref)
        try:
            ref_data = download_and_clean_resource(ref_url)
            clean_refs.append(ref_data)
            logger.info(f"Successfully processed reference: {ref_url}")
        except Exception as e:
            logger.error(f"Failed to download reference {ref_url}: {str(e)}")

    kg["downloadedRefs"] = clean_refs

    # 3) Save knowledge graph
    with open(output_kg, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote knowledge graph to {output_kg}")

    # 4) Create Dolly dataset
    dataset = []
    items = list(kg["classes"].values()) + list(kg["properties"].values())
    items = items[:limit]

    for idx, item in enumerate(items, start=1):
        label = item["label"]
        definition = item["definition"]
        knowledge_context = f"{label}. {definition}"

        logger.info(f"[{idx}/{limit}] Generating questions for: {label}")
        # a) Ask questioner agent
        questioner_prompt_result = questioner_agent_invoke(knowledge_context)

        # b) Check question array
        question_checker_result = question_checker_agent_invoke(questioner_prompt_result)
        try:
            final_questions = json.loads(question_checker_result)
            if not isinstance(final_questions, list):
                final_questions = []
        except:
            final_questions = []

        # c) For each question, maker -> formatter -> checker
        for q in final_questions:
            raw_answer = maker_agent(q, knowledge_context)
            draft_json = formatter_agent(raw_answer, q, knowledge_context)
            checked_json = checker_agent(draft_json)
            # parse
            try:
                final_data = json.loads(checked_json)
                if isinstance(final_data, dict):
                    dataset.append(final_data)
                elif isinstance(final_data, list):
                    dataset.extend(final_data)
            except:
                logger.warning(f"Failed to parse final JSON for question: {q}")

    # 5) Save dataset
    with open(output_ds, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Dolly Q&A dataset to {output_ds}")

    return dataset
