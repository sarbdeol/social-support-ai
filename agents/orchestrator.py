from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from agents.extractor   import extract_data
from agents.validator   import validate_data
from agents.eligibility import check_eligibility
from agents.decision    import make_decision


# ── STATE DEFINITION ──────────────────────────────────
class AgentState(TypedDict):
    applicant_raw:  dict
    extracted_data: dict
    validation:     dict
    ml_result:      dict
    final_output:   dict
    messages:       List[str]


# ── BUILD GRAPH ───────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extractor",   extract_data)
    graph.add_node("validator",   validate_data)
    graph.add_node("eligibility", check_eligibility)
    graph.add_node("decision",    make_decision)

    graph.set_entry_point("extractor")
    graph.add_edge("extractor",   "validator")
    graph.add_edge("validator",   "eligibility")
    graph.add_edge("eligibility", "decision")
    graph.add_edge("decision",    END)

    return graph.compile()


pipeline = build_graph()


# ── MAIN ENTRY POINT ──────────────────────────────────
def run_pipeline(applicant_data: dict) -> dict:
    initial_state = {
        "applicant_raw":  applicant_data,
        "extracted_data": {},
        "validation":     {},
        "ml_result":      {},
        "final_output":   {},
        "messages":       []
    }

    result = pipeline.invoke(initial_state)
    return result