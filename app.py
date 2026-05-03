"""
Prototype: AI Feedback Quality Study
Master's thesis - baseline vs. criteria-enhanced prompting
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
import gspread
from google.oauth2.service_account import Credentials


# ------------------------------------------------------------
# Basic configuration
# ------------------------------------------------------------

MODEL_NAME = "gpt-4o-mini-2024-07-18"
MODEL_TEMPERATURE = 0.2
MAX_TOKENS = 350

DATA_PHASE = "main_study"
PROMPT_VERSION = "v2"
APP_VERSION = "2026-05-03"

DATA_FILE = Path("responses.json")
MATERIALS_DIR = Path("study_materials")


# ------------------------------------------------------------
# Task setup
# ------------------------------------------------------------

TASKS = {
    "count_vowels": {
        "title": "Count the Vowels",
        "function_name": "count_vowels",
        "placeholder": "def count_vowels(text):\n    ...",
        "task_description": "task_description.md",
        "reference_implementation": "reference_implementation.py",
        "reference_rationale": "reference_implementation_rationale.md",
        "software_criteria": "software_assessment_criteria.md",
        "test_cases": "test_cases.json",
    },
    "find_max": {
        "title": "Find the Maximum",
        "function_name": "find_max",
        "placeholder": "def find_max(numbers):\n    ...",
        "task_description": "task_description_find_max.md",
        "reference_implementation": "reference_implementation_find_max.py",
        "reference_rationale": "reference_implementation_rationale_find_max.md",
        "software_criteria": "software_assessment_criteria_find_max.md",
        "test_cases": "test_cases_find_max.json",
    },
}


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


FEEDBACK_QUALITY_CRITERIA = read_text_file(
    MATERIALS_DIR / "feedback_quality_criteria.md"
)
BASELINE_PROMPT_TEMPLATE = read_text_file(MATERIALS_DIR / "baseline_prompt.txt")
CRITERIA_ENHANCED_PROMPT_TEMPLATE = read_text_file(
    MATERIALS_DIR / "criteria_enhanced_prompt.txt"
)


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

def get_task_materials(task_id: str) -> dict[str, Any]:
    task = TASKS[task_id]

    return {
        "task_id": task_id,
        "task_title": task["title"],
        "function_name": task["function_name"],
        "placeholder": task["placeholder"],
        "task_description": read_text_file(MATERIALS_DIR / task["task_description"]),
        "reference_implementation": read_text_file(MATERIALS_DIR / task["reference_implementation"]),
        "reference_rationale": read_text_file(MATERIALS_DIR / task["reference_rationale"]),
        "software_criteria": read_text_file(MATERIALS_DIR / task["software_criteria"]),
        "test_cases": read_json_file(MATERIALS_DIR / task["test_cases"]),
    }


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        st.error("OpenAI API key not found.")
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
        return f"The feedback could not be generated at the moment. Error: {error}"


def run_student_tests(
    student_code: str,
    function_name: str,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    test_payload = {
        "student_code": student_code,
        "function_name": function_name,
        "test_cases": test_cases,
    }

    runner_code = r'''
import json
import sys

payload = json.loads(sys.stdin.read())
student_code = payload["student_code"]
function_name = payload["function_name"]
test_cases = payload["test_cases"]

namespace = {}
results = []

try:
    exec(student_code, namespace)
    function = namespace.get(function_name)

    if function is None or not callable(function):
        output = {
            "status": "failed",
            "summary": f"No callable function named {function_name} was defined.",
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
                "total": len(test_cases),
                "results": [],
            }

        return json.loads(completed.stdout)

    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "summary": "The submitted code timed out during testing.",
            "passed": 0,
            "total": len(test_cases),
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


def get_google_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes,
    )

    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(st.secrets["GOOGLE_SHEET_ID"])
    return spreadsheet.sheet1


def save_record(record: dict[str, Any]) -> None:
    try:
        worksheet = get_google_worksheet()

        flat_record = {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in record.items()
        }

        existing_values = worksheet.get_all_values()

        if not existing_values:
            worksheet.append_row(list(flat_record.keys()))

        worksheet.append_row(list(flat_record.values()))

    except Exception as error:
        st.error(f"Google Sheets error: {error}")
        st.stop()


def initialize_session_state() -> None:
    if "step" not in st.session_state:
        st.session_state.step = "intro"

    if "feedback_order" not in st.session_state:
        st.session_state.feedback_order = random.choice(
            ["baseline_first", "enhanced_first"]
        )

    if "participant_id" not in st.session_state:
        st.session_state.participant_id = f"P{random.randint(100000, 999999)}"

    if "task_id" not in st.session_state:
        st.session_state.task_id = random.choice(list(TASKS.keys()))

    if "task_materials" not in st.session_state:
        st.session_state.task_materials = get_task_materials(st.session_state.task_id)


def likert_radio(label: str, key: str) -> int | None:
    return st.radio(
        label,
        options=LIKERT_OPTIONS,
        index=None,
        horizontal=True,
        captions=LIKERT_CAPTIONS,
        key=key,
    )


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(page_title="AI Feedback Study", layout="centered")
initialize_session_state()

task = st.session_state.task_materials


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
        """
    )

    if st.button("Start"):
        st.session_state.step = "task"
        st.rerun()


elif st.session_state.step == "task":
    st.title(task["task_title"])
    st.markdown(task["task_description"])
    st.markdown("---")

    student_code = st.text_area(
        "Write your Python code here:",
        height=260,
        placeholder=task["placeholder"],
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
    st.markdown(f"**Task:** {task['task_title']}")

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

        test_results = run_student_tests(
            student_code=student_code,
            function_name=task["function_name"],
            test_cases=task["test_cases"],
        )
        st.session_state.test_results = test_results

        baseline_prompt = BASELINE_PROMPT_TEMPLATE.format(
            task=task["task_description"],
            code=student_code,
        )

        enhanced_prompt = CRITERIA_ENHANCED_PROMPT_TEMPLATE.format(
            task=task["task_description"],
            code=student_code,
            reference_rationale=task["reference_rationale"],
            reference_solution=task["reference_implementation"],
            test_results=format_test_results_for_prompt(test_results),
            software_criteria=task["software_criteria"],
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
    st.markdown(f"**Task:** {task['task_title']}")

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
    st.markdown(f"**Task:** {task['task_title']}")

    st.markdown(
        """
        Please evaluate each feedback version against the following **feedback quality characteristics**.

        Consider whether the feedback is understandable, accurate, specific, actionable, useful, and pedagogically helpful for learning.

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
        index=None,
    )

    perceived_difference = st.text_area(
        "Briefly explain the most important difference between Feedback A and Feedback B.",
        height=120,
    )

    missing_or_problematic = st.text_area(
        "Were there any aspects of either feedback version that were particularly helpful, constructive, unclear, misleading, too vague, too complex, or potentially unhelpful?",
        height=120,
    )

    participant_comment = st.text_area(
        "Any additional comments? (optional)",
        height=120,
    )

    if st.button("Submit evaluation"):
        if any(value is None for value in ratings_a.values()) or any(value is None for value in ratings_b.values()):
            st.warning("Please rate all feedback statements before submitting.")
            st.stop()

        if preferred_feedback is None:
            st.warning("Please select an overall feedback preference before submitting.")
            st.stop()

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
            "task_id": task["task_id"],
            "task_title": task["task_title"],
            "function_name": task["function_name"],
            "task_description": task["task_description"],
            "reference_implementation_rationale": task["reference_rationale"],
            "reference_implementation": task["reference_implementation"],
            "software_assessment_criteria": task["software_criteria"],
            "feedback_quality_criteria": FEEDBACK_QUALITY_CRITERIA,
            "test_cases": task["test_cases"],
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