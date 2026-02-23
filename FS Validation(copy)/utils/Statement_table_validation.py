import os
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# ---------------------------
# 1. Load environment variables
# ---------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
base_url = "https://api.groq.com/openai/v1"

# ---------------------------
# 2. Define Output Schemas
# ---------------------------
class TableSummary(BaseModel):
    Table_Name: str = Field(..., alias="Table Name")
    Status: str

class TableErrorDetail(BaseModel):
    Table_Name: str = Field(..., alias="Table Name")
    Row_Index: int = Field(..., alias="Row Index")
    Column: str
    Expected_Value: float = Field(..., alias="Expected Value")
    Found_Value: float = Field(..., alias="Found Value")

class ValidationOutput(BaseModel):
    summary: List[TableSummary]
    details: List[TableErrorDetail]

# ---------------------------
# 3. Output Parser and Prompt
# ---------------------------
output_parser = PydanticOutputParser(pydantic_object=ValidationOutput)

prompt = PromptTemplate(
    template="""
You are a financial table validation expert.

Given a table named {table_name} with:
- The first column as row labels (e.g., Revenue, Total)
- Remaining columns as numeric values (financial periods)
- Some rows represent calculations (totals, subtotals, net change)

Tasks:
1. Identify rows that represent calculations.
2. Validate these rows using addition/subtraction logic across columns.
3. Ignore formatting (e.g., commas).
4. Report mismatches with expected vs. found values.

If all values are correct:
- Return one summary with "Status": "Correct", no details.

If any errors:
- Return one summary with "Status": "Wrong"
- Include details: row index, column, expected vs. found value.

Output must follow this JSON format:
{format_instructions}

--- Table Data ---
{table_markdown}
""",
    input_variables=["table_name", "table_markdown"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)


# ---------------------------
# 4. Initialize LLM
# ---------------------------
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=base_url,
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0
)

# ---------------------------
# 5. Create LangChain Chain
# ---------------------------
chain = (
    {
        "table_name": lambda x: x["table_name"],
        "table_markdown": lambda x: x["df"].fillna("").to_markdown(index=False)
    }
    | prompt
    | llm
    | output_parser
)

# ---------------------------
# 6. Validate All Tables
# ---------------------------
def statement__tables_validation(final_subtables: Dict[str, pd.DataFrame]):
    all_summary = []
    all_details = []

    for name, df in final_subtables.items():
        try:
            result = chain.invoke({"table_name": name, "df": df})
            display
            all_summary.extend(result.summary)
            all_details.extend(result.details)
        except Exception as e:
            print(f"❌ Validation failed for '{name}': {str(e)}")
            all_summary.append(TableSummary(Table_Name=name, Status=f"Error: {str(e)}"))

    summary_df = pd.DataFrame([s.dict(by_alias=True) for s in all_summary])
    details_df = pd.DataFrame([d.dict(by_alias=True) for d in all_details]) if all_details else pd.DataFrame()

    return summary_df, details_df


