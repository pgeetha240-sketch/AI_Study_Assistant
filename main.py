from flask import Flask, render_template, request, redirect, session
from google import genai
from google.genai import types

import os
import json
import re
import ast
import operator


app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "ai-study-assistant-secret-key"
)

GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def normalize_text(text):
    return " ".join(
        text.strip().lower().split()
    )


def ask_gemini(prompt, temperature=0.2):
    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Please set your Gemini API key and reopen the terminal."
        )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )

    answer = response.text

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer.strip()


def clean_ai_json(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


# --------------------------------------------------
# SAFE PYTHON EVALUATOR
# --------------------------------------------------

SAFE_FUNCTIONS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool
}


BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}


UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_
}


def safe_eval_node(node, variables=None):

    if variables is None:
        variables = {}

    if isinstance(node, ast.Constant):
        if isinstance(
            node.value,
            (str, int, float, bool, type(None))
        ):
            return node.value

        raise ValueError("Unsupported constant")

    if isinstance(node, ast.List):
        return [
            safe_eval_node(element, variables)
            for element in node.elts
        ]

    if isinstance(node, ast.Tuple):
        return tuple(
            safe_eval_node(element, variables)
            for element in node.elts
        )

    if isinstance(node, ast.Name):

        if node.id in variables:
            return variables[node.id]

        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]

        raise ValueError(
            f"Unknown variable: {node.id}"
        )

    if isinstance(node, ast.BinOp):

        operation = BIN_OPS.get(
            type(node.op)
        )

        if operation is None:
            raise ValueError(
                "Unsupported operator"
            )

        left = safe_eval_node(
            node.left,
            variables
        )

        right = safe_eval_node(
            node.right,
            variables
        )

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):

        operation = UNARY_OPS.get(
            type(node.op)
        )

        if operation is None:
            raise ValueError(
                "Unsupported unary operator"
            )

        value = safe_eval_node(
            node.operand,
            variables
        )

        return operation(value)

    if isinstance(node, ast.Compare):

        left = safe_eval_node(
            node.left,
            variables
        )

        for operation_node, comparator_node in zip(
            node.ops,
            node.comparators
        ):

            right = safe_eval_node(
                comparator_node,
                variables
            )

            if isinstance(
                operation_node,
                ast.Eq
            ):
                result = left == right

            elif isinstance(
                operation_node,
                ast.NotEq
            ):
                result = left != right

            elif isinstance(
                operation_node,
                ast.Lt
            ):
                result = left < right

            elif isinstance(
                operation_node,
                ast.LtE
            ):
                result = left <= right

            elif isinstance(
                operation_node,
                ast.Gt
            ):
                result = left > right

            elif isinstance(
                operation_node,
                ast.GtE
            ):
                result = left >= right

            else:
                raise ValueError(
                    "Unsupported comparison"
                )

            if not result:
                return False

            left = right

        return True

    if isinstance(node, ast.Call):

        if not isinstance(
            node.func,
            ast.Name
        ):
            raise ValueError(
                "Unsupported function"
            )

        function_name = node.func.id

        if function_name not in SAFE_FUNCTIONS:
            raise ValueError(
                "Unsafe function"
            )

        function = SAFE_FUNCTIONS[
            function_name
        ]

        arguments = [
            safe_eval_node(
                argument,
                variables
            )
            for argument in node.args
        ]

        return function(*arguments)

    if isinstance(node, ast.Subscript):

        value = safe_eval_node(
            node.value,
            variables
        )

        index = safe_eval_node(
            node.slice,
            variables
        )

        return value[index]

    raise ValueError(
        "Unsupported Python expression"
    )


def safe_python_execute(code):

    variables = {}
    output = []

    tree = ast.parse(
        code,
        mode="exec"
    )

    for statement in tree.body:

        if isinstance(
            statement,
            ast.Assign
        ):

            value = safe_eval_node(
                statement.value,
                variables
            )

            for target in statement.targets:

                if not isinstance(
                    target,
                    ast.Name
                ):
                    raise ValueError(
                        "Unsupported assignment"
                    )

                variables[
                    target.id
                ] = value

        elif isinstance(
            statement,
            ast.Expr
        ):

            if isinstance(
                statement.value,
                ast.Call
            ):

                if (
                    isinstance(
                        statement.value.func,
                        ast.Name
                    )
                    and statement.value.func.id == "print"
                ):

                    values = [
                        safe_eval_node(
                            argument,
                            variables
                        )
                        for argument
                        in statement.value.args
                    ]

                    output.append(
                        " ".join(
                            str(value)
                            for value in values
                        )
                    )

                else:
                    safe_eval_node(
                        statement.value,
                        variables
                    )

            else:
                safe_eval_node(
                    statement.value,
                    variables
                )

        else:

            raise ValueError(
                "Unsupported Python statement"
            )

    return "\n".join(output)


def python_execution_result(code):

    try:

        result = safe_python_execute(
            code
        )

        return {
            "success": True,
            "output": result
        }

    except Exception as error:

        return {
            "success": False,
            "output": str(error)
        }


# --------------------------------------------------
# QUIZ HELPERS
# --------------------------------------------------

def extract_code(question):

    matches = re.findall(
        r"```(?:python)?\s*(.*?)```",
        question,
        flags=re.IGNORECASE | re.DOTALL
    )

    if matches:
        return matches[0].strip()

    patterns = [
        r"\bprint\s*\(.*?\)",
        r"\b[a-zA-Z_]\w*\s*=\s*[^\n,]+"
    ]

    found = []

    for pattern in patterns:

        found.extend(
            re.findall(
                pattern,
                question
            )
        )

    if found:
        return "\n".join(found)

    return ""


def extract_literal(question):

    string_match = re.search(
        r"""['"]([^'"]*)['"]""",
        question
    )

    if string_match:
        return string_match.group(1)

    number_match = re.search(
        r"\b\d+(?:\.\d+)?\b",
        question
    )

    if number_match:
        return number_match.group(0)

    return None


def extract_math_expression(question):

    patterns = [
        r"(\d+(?:\s*[+\-*/]\s*\d+)+)",
        r"(\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question
        )

        if match:
            return match.group(1)

    return ""


def validate_quiz_question(question_data):

    if not isinstance(
        question_data,
        dict
    ):
        return False

    required_fields = [
        "question",
        "options",
        "answer"
    ]

    for field in required_fields:

        if field not in question_data:
            return False

    question = question_data[
        "question"
    ]

    options = question_data[
        "options"
    ]

    answer = question_data[
        "answer"
    ]

    if not isinstance(
        question,
        str
    ):
        return False

    if not isinstance(
        options,
        list
    ):
        return False

    if len(options) != 4:
        return False

    if not isinstance(
        answer,
        str
    ):
        return False

    if answer not in options:
        return False

    for option in options:

        if not isinstance(
            option,
            str
        ):
            return False

        if not option.strip():
            return False

    return True


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    if "history" not in session:
        session["history"] = []

    answer = ""
    question = ""

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        subject = request.form.get(
            "subject",
            "General"
        )

        answer_type = request.form.get(
            "answer_type",
            "Detailed"
        )

        if question:

            question_normalized = normalize_text(
                question
            )

            # Reliable answer for this common Tamil question
            fixed_answers = {

                "python என்றால் என்ன?":
                    "Python என்பது ஒரு programming language. இது software, websites, automation, data science மற்றும் பல்வேறு applications உருவாக்க பயன்படுத்தப்படுகிறது."

            }

            if question_normalized in fixed_answers:

                answer = fixed_answers[
                    question_normalized
                ]

            else:

                prompt = f"""
You are an accurate and helpful AI Study Assistant.

Subject: {subject}

Answer type: {answer_type}

Student question:
{question}

Give a clear, accurate and easy-to-understand answer.

LANGUAGE RULES:
- Reply in the same language used by the student.
- If the student asks in Tamil script, answer in simple natural Tamil.
- If the student asks in Tanglish, answer in simple Tanglish.
- If the student asks in English, answer in English.
- Do not unnecessarily translate the student's question.
- Do not mix languages unless a technical term needs to remain in English.

TAMIL ACCURACY RULES:
- Use natural and meaningful Tamil.
- Do not invent Tamil words.
- For technical subjects, use the correct English technical term when necessary.
- If a technical term does not have a commonly understood Tamil equivalent, keep the English technical term.
- Never describe Python incorrectly.
- Python is a programming language.

PYTHON RULES:
- Follow real Python behavior.
- Do not invent Python behavior.
- If code is provided, carefully determine the actual output or error.
- Explain code step by step when useful.
- Never claim that assigning a string to a variable automatically causes an error unless the actual code does so.

MATHEMATICS RULES:
- Calculate carefully.
- Show the important steps for detailed answers.
- Give the final answer clearly.

GENERAL RULES:
- Do not make up facts.
- If you are uncertain, say that you are uncertain instead of inventing an answer.
- Keep the answer appropriate for a student.
- Use simple explanations.

If the answer type is Short:
Give a concise answer.

If the answer type is Detailed:
Give a clear explanation with useful details.
"""

                try:

                    answer = ask_gemini(
                        prompt,
                        temperature=0.2
                    )

                except Exception as error:

                    answer = (
                        "AI Error: "
                        + str(error)
                    )

            history = session.get(
                "history",
                []
            )

            history.append(
                {
                    "question": question,
                    "answer": answer,
                    "subject": subject
                }
            )

            session["history"] = history[-10:]

            session.modified = True

    return render_template(
        "index.html",
        answer=answer,
        question=question,
        history=session.get(
            "history",
            []
        )
    )


# --------------------------------------------------
# CLEAR HISTORY
# --------------------------------------------------

@app.route(
    "/clear",
    methods=["GET", "POST"]
)
def clear_history():

    session["history"] = []

    session.modified = True

    return redirect("/")


# --------------------------------------------------
# QUIZ GENERATOR
# --------------------------------------------------

@app.route(
    "/quiz",
    methods=["GET", "POST"]
)
def quiz():

    questions = []

    if request.method == "POST":

        subject = request.form.get(
            "subject",
            "Python"
        )

        topic = request.form.get(
            "topic",
            ""
        ).strip()

        count_text = request.form.get(
            "count",
            "5"
        )

        try:
            count = int(count_text)
        except ValueError:
            count = 5

        count = max(
            1,
            min(count, 10)
        )

        prompt = f"""
Create {count} accurate multiple-choice quiz questions.

Subject:
{subject}

Topic:
{topic}

Return ONLY valid JSON.

The JSON must have this exact structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option 1",
        "Option 2",
        "Option 3",
        "Option 4"
      ],
      "answer": "The exact correct option"
    }}
  ]
}}

Rules:

1. Every question must have exactly 4 options.
2. Only one option must be correct.
3. The answer must exactly match one option.
4. Questions must test real factual knowledge.
5. Do not invent facts.
6. For Python questions, follow actual Python behavior.
7. If asking about Python code output, include the complete code inside the question.
8. Never claim that assigning a string to an integer variable automatically produces a TypeError.
9. Never claim that assigning a string to a list variable automatically produces a TypeError.
10. Do not create ambiguous questions.
11. Do not use "all of the above".
12. Make incorrect options plausible.
"""

        try:

            raw_response = ask_gemini(
                prompt,
                temperature=0.2
            )

            cleaned = clean_ai_json(
                raw_response
            )

            data = json.loads(
                cleaned
            )

            generated_questions = data.get(
                "questions",
                []
            )

            for item in generated_questions:

                if validate_quiz_question(
                    item
                ):

                    questions.append(
                        item
                    )

                if len(questions) >= count:
                    break

        except Exception:
            questions = []

    return render_template(
        "quiz.html",
        questions=questions
    )


# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )