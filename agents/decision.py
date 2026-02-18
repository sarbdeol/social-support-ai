from langchain_community.llms import Ollama
import json

llm = Ollama(model="llama3")

ENABLEMENT_OPTIONS = [
    "Job Matching — connect with relevant job openings based on your skills",
    "Vocational Training — free government-sponsored skills courses",
    "Career Counseling — one-on-one guidance sessions with career advisors",
    "Micro-Finance Support — small business loans for self-employment",
    "Skills Upskilling — digital literacy and professional development programs"
]

def make_decision(state: dict) -> dict:
    """
    Generates final decision with explanation + economic enablement recommendations.
    Uses ReAct reasoning: Think → Act → Observe pattern.
    """
    data       = state.get("extracted_data", {})
    ml_result  = state.get("ml_result", {})
    validation = state.get("validation", {})

    decision   = ml_result.get("decision", "DECLINE")
    confidence = ml_result.get("confidence", 0)
    flags      = validation.get("flags", [])

    prompt = f"""
You are a social support decision agent for Abu Dhabi government.
Based on the assessment below, provide a final decision with explanation
and recommend economic enablement programs.

Applicant Profile:
{json.dumps(data, indent=2)}

ML Model Decision: {decision} (Confidence: {confidence}%)
Validation Flags: {flags}

Available Enablement Programs:
{json.dumps(ENABLEMENT_OPTIONS, indent=2)}

Think step by step (ReAct):
1. THINK: What are the key factors in this case?
2. ACT: What is the final decision and why?
3. OBSERVE: Which 2-3 enablement programs best fit this applicant?

Return ONLY this JSON:
{{
    "final_decision": "{decision}",
    "decision_reason": <2-3 sentence explanation for the applicant>,
    "enablement_recommendations": [<2-3 program names from the list>],
    "next_steps": <one sentence on what applicant should do next>
}}
"""
    response = llm.invoke(prompt)

    try:
        start  = response.find('{')
        end    = response.rfind('}') + 1
        output = json.loads(response[start:end])
    except Exception:
        output = {
            "final_decision":             decision,
            "decision_reason":            f"Based on your financial profile, your application has been {decision.lower()}ed with {confidence}% confidence.",
            "enablement_recommendations": ENABLEMENT_OPTIONS[:2],
            "next_steps":                 "Please visit your nearest social support center for further assistance."
        }

    state["final_output"] = output
    state["messages"].append(
        f"✅ Decision complete — {output['final_decision']} | "
        f"{len(output.get('enablement_recommendations', []))} enablement programs recommended"
    )
    return state