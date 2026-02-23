import os
import re
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
base_url = "https://api.groq.com/openai/v1"

# --- Initialize LLM ---
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=base_url,
    model_name="llama3-70b-8192"
)

# --- Define Response Schema ---
response_schemas = [
    ResponseSchema(name="status", description="correct or wrong"),
    ResponseSchema(name="errors", description="List of dictionaries with row_index, expected, and found values")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# --- Prompt Templates ---
system_template = (
    "You are a qualified financial auditor. "
    "Check the mathematical correctness of totals in the given financial table. "
    "Output must follow the provided format exactly."
)

human_template = """
Table Name: {table_name}

Validation Rules:
- ✅ Only check calculations (ignore formatting like commas).
- ❌ Do not report correct rows.
- ✅ Report rows where Expected ≠ Found (numerically).
- ⚠️ If everything is correct, say so.
❗ Only include incorrect rows where numbers do not match after removing commas.

Format Instructions:
{format_instructions}

--- Table Data ---
{table_markdown}
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

# --- Safe Arithmetic Eval ---
def safe_eval(expr: str) -> float:
    try:
        return eval(expr, {"__builtins__": None}, {})
    except:
        return None

# --- Table Validation Function ---
def validate_table(table_name: str, df: pd.DataFrame):
    table_markdown = df.to_markdown(index=True, tablefmt="grid")

    messages = prompt_template.format_messages(
        table_name=table_name,
        table_markdown=table_markdown,
        format_instructions=format_instructions
    )

    response = llm.invoke(messages)
    result = response.content.strip()

    errors = []
    status = "Unknown"

    try:
        parsed = parser.parse(result)
        status = parsed["status"].capitalize()

        for err in parsed["errors"]:
            row_idx = int(err.get("row_index", -1))
            expected_raw = str(err.get("expected", ""))
            found_raw = str(err.get("found", ""))

            expected = safe_eval(expected_raw.replace(",", ""))
            found = safe_eval(found_raw.replace(",", ""))

            if expected is None or found is None:
                continue

            if abs(expected - found) > 1e-2:
                errors.append({
                    "Table Name": table_name,
                    "Row Index": row_idx,
                    "Expected Value": expected,
                    "Found Value": found
                })

    except Exception as e:
        print(f"⚠️ Structured parsing failed: {e}")
        print("⚠️ Raw LLM output:")
        print(result)

        status_match = re.search(r"Status\s*:\s*(correct|wrong)", result, re.IGNORECASE)
        status = status_match.group(1).capitalize() if status_match else "Unknown"

        if status.lower() == "wrong":
            row_pattern = r"(\d+)\s*[:,\-]?\s*([-+*/\d, ()]+)\s*[:=>\-]+?\s*([-+*/\d, ()]+)"
            for match in re.finditer(row_pattern, result):
                try:
                    idx = int(match.group(1))
                    expected_raw = match.group(2).strip()
                    found_raw = match.group(3).strip()

                    expected = safe_eval(expected_raw.replace(",", ""))
                    found = safe_eval(found_raw.replace(",", ""))

                    if expected is None or found is None:
                        continue

                    if abs(expected - found) > 1e-2:
                        errors.append({
                            "Table Name": table_name,
                            "Row Index": idx,
                            "Expected Value": expected,
                            "Found Value": found
                        })
                except Exception as fallback_err:
                    print(f"⚠️ Eval fallback failed: {fallback_err}")
                    continue

    return {"Table Name": table_name, "Status": "Correct" if not errors else "Wrong"}, errors

# --- Full Validation Pipeline ---
def run_validation(final_subtables: dict):
    summary = []
    all_errors = []

    for table_name, df in final_subtables.items():
        status, errors = validate_table(table_name, df)
        summary.append(status)
        all_errors.extend(errors)

    summary_df = pd.DataFrame(summary)
    print("\n✅ Summary Report:")
    display(summary_df)

    if all_errors:
        error_df = pd.DataFrame(all_errors)[[
            "Table Name", "Row Index", "Expected Value", "Found Value"
        ]]
        print("\n❌ Detailed Error Report:")
        display(error_df)
    else:
        print("\n🎉 No errors found in any table!")
