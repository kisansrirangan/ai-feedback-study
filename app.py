"""
Prototype: AI Feedback Quality Study
Master's thesis - baseline vs. criteria-enhanced prompting

Run locally:
    python3 -m streamlit run app.py

The app requires an OpenAI API key, either:
1. as an environment variable: OPENAI_API_KEY
2. or in .streamlit/secrets.toml
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI


# ------------------------------------------------------------
# Basic configuration
# ------------------------------------------------------------

MODEL_NAME = "gpt-4o-mini-2024-07-18"
MODEL_TEMPERATURE = 0.2
MAX_TOKENS = 350

DATA_PHASE = "main_study"
PROMPT_VERSION = "v2"
APP_VERSION = "2026-04-29"

DATA_FILE = Path("responses.json")
MATERIALS_DIR = Path("study_materials")

TASK_ID = "count_vowels"
TASK_TITLE = "Count the Vowels"
TASK_PLACEHOLDER = "def count_vowels(text):\n    ..."


# ------------------------------------------------------------
# Load study materials
# ------------------------------------------------------------

def read_text_file(path: Path) -> str:
    if not path.exists():
        st.error(f"Missing required study material file: {path}")
        st.stop()
    return path.read_text(encoding="utf-8").strip()


def read_json_file(path: Path) -> Any:
    if not path.exists():
        st.error(f"Missing required study material file: {path}")
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


TASK_DESCRIPTION = read_text_file(MATERIALS_DIR / "task_description.md")
REFERENCE_IMPLEMENTATION = read_text_file(MATERIALS_DIR / "reference_implementation.py")
REFERENCE_IMPLEMENTATION_RATIONALE = read_text_file(
    MATERIALS_DIR / "reference_implementation_rationale.md"
)
SOFTWARE_ASSESSMENT_CRITERIA = read_text_file(
    MATERIALS_DIR / "software_assessment_criteria.md"
)
FEEDBACK_QUALITY_CRITERIA = read_text_file(
    MATERIALS_DIR / "feedback_quality_criteria.md"
)
BASELINE_PROMPT_TEMPLATE = read_text_file(MATERIALS_DIR / "baseline_prompt.txt")
CRITERIA_ENHANCED_PROMPT_TEMPLATE = read_text_file(
    MATERIALS_DIR / "criteria_enhanced_prompt.txt"
)
TEST_CASES = read_json_file(MATERIALS_DIR / "test_cases.json")


# ------------------------------------------------------------
# Questionnaire statements
# ------------------------------------------------------------

RATING_DIMENSIONS = [
    ("understandability", "The feedback is easy to understand."),
    ("accuracy", "The feedback correctly identifies real strengths or weaknesses in the code."),
    ("specificity", "The feedback refers to specific aspects of the submitted code."),
    ("actionability", "The feedback makes it clear what the student should improve next."),
    ("usefulness", "The feedback would help the student improve their programming solution."),
    ("pedagogical_value", "The feedback supports learning rather than only judging the answer."),
]

LIKERT_OPTIONS = [1, 2, 3, 4, 5]
LIKERT_CAPTIONS = [
    "1 = Strongly disagree",
    "2 = Disagree",
    "3 = Neutral",
    "4 = Agree",
    "5 = Strongly agree",
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        st.error(
            "OpenAI API key not found. Add it either as an environment variable "
            "or in `.streamlit/secrets.toml`."
        )
        st.stop()

    return OpenAI(api_key=api_key)


def generate_feedback(client: OpenAI, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=MODEL_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    except Exception as error:
        return (
            "The feedback could not be generated at the moment. "
            f"Error: {error}"
        )


def run_student_tests(student_code: str) -> dict[str, Any]:
    test_payload = {
        "student_code": student_code,
        "test_cases": TEST_CASES,
    }

    runner_code = r'''
import json
import sys

payload = json.loads(sys.stdin.read())
student_code = payload["student_code"]
test_cases = payload["test_cases"]

namespace = {}
results = []

try:
    exec(student_code, namespace)
    function = namespace.get("count_vowels")

    if function is None or not callable(function):
        output = {
            "status": "failed",
            "summary": "No callable function named count_vowels was defined.",
            "passed": 0,
            "total": len(test_cases),
            "results": [],
        }
    else:
        passed = 0
        for case in test_cases:
            try:
                actual = function(case["input"])
                ok = actual == case["expected"]
                if ok:
                    passed += 1
                results.append({
                    "input": case["input"],
                    "expected": case["expected"],
                    "actual": actual,
                    "passed": ok,
                    "reason": case.get("reason", ""),
                })
            except Exception as e:
                results.append({
                    "input": case["input"],
                    "expected": case["expected"],
                    "actual": f"Error: {e}",
                    "passed": False,
                    "reason": case.get("reason", ""),
                })

        output = {
            "status": "completed",
            "summary": f"Passed {passed} of {len(test_cases)} tests.",
            "passed": passed,
            "total": len(test_cases),
            "results": results,
        }

except Exception as e:
    output = {
        "status": "failed",
        "summary": f"The submitted code could not be executed: {e}",
        "passed": 0,
        "total": len(test_cases),
        "results": [],
    }

print(json.dumps(output, ensure_ascii=False))
'''

    try:
        completed = subprocess.run(
            [sys.executable, "-c", runner_code],
            input=json.dumps(test_payload),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if completed.returncode != 0:
            return {
                "status": "failed",
                "summary": completed.stderr.strip() or "The test runner failed.",
                "passed": 0,
                "total": len(TEST_CASES),
                "results": [],
            }

        return json.loads(completed.stdout)

    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "summary": "The submitted code timed out during testing.",
            "passed": 0,
            "total": len(TEST_CASES),
            "results": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "summary": f"Could not run tests: {error}",
            "passed": 0,
            "total": len(TEST_CASES),
            "results": [],
        }


def format_test_results_for_prompt(test_results: dict[str, Any]) -> str:
    lines = [test_results.get("summary", "No test summary available.")]

    for result in test_results.get("results", []):
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(
            f"- {status}: input={result['input']!r}, expected={result['expected']!r}, "
            f"actual={result['actual']!r}. Reason: {result.get('reason', '')}"
        )

    return "\n".join(lines)


def load_existing_records() -> list[dict[str, Any]]:
    if not DATA_FILE.exists():
        return []

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_record(record: dict[str, Any]) -> None:
    records = load_existing_records()
    records.append(record)

    backup_file = DATA_FILE.with_suffix(".backup.json")

    if DATA_FILE.exists():
        backup_file.write_text(
            DATA_FILE.read_text(encoding="utf-8"),
            encoding="utf-8"
        )

    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)


def initialize_session_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = "intro"

    if "feedback_order" not in st.session_state:
        st.session_state.feedback_order = random.choice(
            ["baseline_first", "enhanced_first"]
        )

    if "participant_id" not in st.session_state:
        st.session_state.participant_id = f"P{random.randint(100000, 999999)}"


def likert_radio(label: str, key: str) -> int:
    return st.radio(
        label,
        options=LIKERT_OPTIONS,
        index=2,
        horizontal=True,
        captions=LIKERT_CAPTIONS,
        key=key,
    )


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(page_title="AI Feedback Study", layout="centered")
initialize_session_state()


# ------------------------------------------------------------
# App flow
# ------------------------------------------------------------

if st.session_state.step == "intro":
    st.title("AI Feedback Study")

    st.markdown(
        """
        Welcome. In this study, you will:

        1. Read a short programming task  
        2. Write your solution in Python  
        3. Reflect briefly on your own solution before receiving feedback  
        4. Receive two feedback versions on your code  
        5. Evaluate both feedback versions in a short questionnaire  

        The study compares two feedback conditions generated by the same AI model:

        - **Feedback condition 1:** minimal prompt without explicit assessment criteria  
        - **Feedback condition 2:** criteria-enhanced prompt with a pedagogical reference implementation, functional test results, software assessment criteria, and feedback quality criteria  

        The same model and settings are used for both feedback conditions. The order of the two feedback versions is randomized.
        Your participation is anonymous.
        """
    )

    if st.button("Start"):
        st.session_state.step = "task"
        st.rerun()


elif st.session_state.step == "task":
    st.title(TASK_TITLE)
    st.markdown(TASK_DESCRIPTION)
    st.markdown("---")

    student_code = st.text_area(
        "Write your Python code here:",
        height=260,
        placeholder=TASK_PLACEHOLDER,
    )

    if st.button("Submit code"):
        if len(student_code.strip()) < 10:
            st.warning("Please write at least a few lines of code before submitting.")
        else:
            st.session_state.student_code = student_code.strip()
            st.session_state.step = "reflection"
            st.rerun()


elif st.session_state.step == "reflection":
    st.title("Before receiving feedback")
    st.markdown(f"**Task:** {TASK_TITLE}")

    st.markdown("**Your submitted code:**")
    st.code(st.session_state.student_code, language="python")

    reflection = st.text_area(
        "Before seeing the AI feedback, what do you think are the main strengths, weaknesses, or uncertainties in your solution?",
        height=150,
        placeholder="Write a short reflection here...",
    )

    if st.button("Continue"):
        st.session_state.pre_feedback_reflection = reflection.strip()
        st.session_state.step = "generating"
        st.rerun()


elif st.session_state.step == "generating":
    st.title("Generating feedback...")

    with st.spinner("Please wait while the feedback is generated."):
        client = get_openai_client()
        student_code = st.session_state.student_code

        test_results = run_student_tests(student_code)
        st.session_state.test_results = test_results

        baseline_prompt = BASELINE_PROMPT_TEMPLATE.format(
            task=TASK_DESCRIPTION,
            code=student_code,
        )

        enhanced_prompt = CRITERIA_ENHANCED_PROMPT_TEMPLATE.format(
            task=TASK_DESCRIPTION,
            code=student_code,
            reference_rationale=REFERENCE_IMPLEMENTATION_RATIONALE,
            reference_solution=REFERENCE_IMPLEMENTATION,
            test_results=format_test_results_for_prompt(test_results),
            software_criteria=SOFTWARE_ASSESSMENT_CRITERIA,
            feedback_criteria=FEEDBACK_QUALITY_CRITERIA,
        )

        baseline_feedback = generate_feedback(client=client, prompt=baseline_prompt)
        enhanced_feedback = generate_feedback(client=client, prompt=enhanced_prompt)

        st.session_state.baseline_prompt = baseline_prompt
        st.session_state.enhanced_prompt = enhanced_prompt
        st.session_state.baseline_feedback = baseline_feedback
        st.session_state.enhanced_feedback = enhanced_feedback

        if st.session_state.feedback_order == "baseline_first":
            st.session_state.feedback_a = baseline_feedback
            st.session_state.feedback_b = enhanced_feedback
            st.session_state.label_a = "baseline_minimal_prompt"
            st.session_state.label_b = "criteria_enhanced_prompt"
        else:
            st.session_state.feedback_a = enhanced_feedback
            st.session_state.feedback_b = baseline_feedback
            st.session_state.label_a = "criteria_enhanced_prompt"
            st.session_state.label_b = "baseline_minimal_prompt"

    st.session_state.step = "feedback"
    st.rerun()


elif st.session_state.step == "feedback":
    st.title("Your feedback")
    st.markdown(f"**Task:** {TASK_TITLE}")

    st.markdown("**Your code:**")
    st.code(st.session_state.student_code, language="python")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Feedback A")
        st.info(st.session_state.feedback_a)

    with col_b:
        st.markdown("### Feedback B")
        st.info(st.session_state.feedback_b)

    if st.button("Continue to evaluation"):
        st.session_state.step = "questionnaire"
        st.rerun()


elif st.session_state.step == "questionnaire":
    st.title("Evaluate the feedback")
    st.markdown(f"**Task:** {TASK_TITLE}")

    st.markdown(
        """
        Please evaluate the two feedback versions as **formative feedback**.

        Formative feedback means feedback intended to help a student understand their current solution and improve their learning, not simply to assign a grade.

        Rate each feedback version on the following statements.  
        **1 = Strongly disagree** and **5 = Strongly agree.**
        """
    )

    st.markdown("---")
    st.markdown("### Feedback A")
    st.info(st.session_state.feedback_a)

    ratings_a: dict[str, int] = {}
    for key, statement in RATING_DIMENSIONS:
        ratings_a[key] = likert_radio(statement, key=f"a_{key}")

    st.markdown("---")
    st.markdown("### Feedback B")
    st.info(st.session_state.feedback_b)

    ratings_b: dict[str, int] = {}
    for key, statement in RATING_DIMENSIONS:
        ratings_b[key] = likert_radio(statement, key=f"b_{key}")

    st.markdown("---")

    preferred_feedback = st.radio(
        "Overall, which feedback version would better support the student’s learning and improvement?",
        ["Feedback A", "Feedback B", "No clear difference"],
        horizontal=True,
    )

    perceived_difference = st.text_area(
        "Briefly explain the most important difference between Feedback A and Feedback B.",
        height=120,
    )

    missing_or_problematic = st.text_area(
        "Was anything missing, misleading, too vague, too complex, or potentially unhelpful in either feedback version?",
        height=120,
    )

    participant_comment = st.text_area(
        "Any additional comments? (optional)",
        height=120,
    )

    if st.button("Submit evaluation"):
        record = {
            "timestamp": datetime.now().isoformat(),
            "participant_id": st.session_state.participant_id,
            "data_phase": DATA_PHASE,
            "prompt_version": PROMPT_VERSION,
            "app_version": APP_VERSION,
            "model_provider": "OpenAI API",
            "model_name": MODEL_NAME,
            "model_temperature": MODEL_TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "task_id": TASK_ID,
            "task_title": TASK_TITLE,
            "task_description": TASK_DESCRIPTION,
            "reference_implementation_rationale": REFERENCE_IMPLEMENTATION_RATIONALE,
            "reference_implementation": REFERENCE_IMPLEMENTATION,
            "software_assessment_criteria": SOFTWARE_ASSESSMENT_CRITERIA,
            "feedback_quality_criteria": FEEDBACK_QUALITY_CRITERIA,
            "test_cases": TEST_CASES,
            "student_code": st.session_state.student_code,
            "pre_feedback_reflection": st.session_state.get("pre_feedback_reflection", ""),
            "test_results": st.session_state.test_results,
            "feedback_order": st.session_state.feedback_order,
            "label_a": st.session_state.label_a,
            "label_b": st.session_state.label_b,
            "baseline_prompt": st.session_state.baseline_prompt,
            "enhanced_prompt": st.session_state.enhanced_prompt,
            "feedback_a": st.session_state.feedback_a,
            "feedback_b": st.session_state.feedback_b,
            "ratings_a": ratings_a,
            "ratings_b": ratings_b,
            "preferred_feedback": preferred_feedback,
            "perceived_difference": perceived_difference,
            "missing_or_problematic": missing_or_problematic,
            "comment": participant_comment,
        }

        save_record(record)
        st.session_state.step = "done"
        st.rerun()

elif st.session_state.step == "done":
    st.title("Thank you")
    st.success("Your response has been recorded.")

    if st.button("Start new response"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()