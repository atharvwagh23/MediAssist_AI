"""One place that creates the chat model (Groq or OpenAI, chosen in .env)."""

from app import config


def get_llm(model: str | None = None):
    """Return a LangChain chat model. The API key is read from the environment.

    model=None uses the default model from .env. The Evaluator passes JUDGE_MODEL here.
    """
    config.validate_llm_config()  # clear error if the key is missing
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or config.GROQ_MODEL, temperature=0)  # temperature=0: stable answers
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model or config.OPENAI_MODEL, temperature=0)
def invoke_structured(prompt_text: str, schema, inputs: dict, model: str | None = None):
    """Run a prompt and return a filled Pydantic `schema` object.

    First try: the model's structured-output (tool call) mode.
    Fallback: some models answer in plain text instead of calling the tool, so we
    ask again for a JSON object and parse it ourselves.
    """
    import re

    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm(model)
    try:
        chain = ChatPromptTemplate.from_template(prompt_text) | llm.with_structured_output(schema)
        return chain.invoke(inputs)
    except Exception:
        keys = ", ".join(schema.model_fields)
        json_prompt = (prompt_text + "\n\nReturn ONLY a JSON object with these keys: " + keys
                       + ". Use true or false for yes/no keys. No other text, no markdown.")
        raw = (ChatPromptTemplate.from_template(json_prompt) | llm).invoke(inputs).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # first JSON object in the reply
        if not match:
            raise ValueError("The model did not return JSON.")
        return schema.model_validate_json(match.group(0))