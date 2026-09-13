import os
import json
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# 1. PROVIDER CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class Provider:
    name: str
    env_var: str
    is_free: bool
    base_url: str | None
    model: str


PROVIDERS = [
    Provider(
        name="OpenAI",
        env_var="OPENAI_API_KEY",
        is_free=False,
        base_url=None,
        model="gpt-4o-mini",
    ),

    # Example of another OpenAI-compatible provider:
    #
    # Provider(
    #     name="Groq",
    #     env_var="GROQ_API_KEY",
    #     is_free=True,
    #     base_url="https://api.groq.com/openai/v1",
    #     model="llama-3.3-70b-versatile",
    # ),
]


# ============================================================
# 2. SELECT PROVIDER
# ============================================================

def select_provider() -> Provider:

    for provider in PROVIDERS:

        api_key = os.getenv(provider.env_var)

        if api_key:
            return provider

    raise RuntimeError(
        "No API key found. Please set an API key "
        "in your environment variables."
    )


# ============================================================
# 3. BUILD OPENAI CLIENT
# ============================================================

def build_client(provider: Provider) -> OpenAI:

    api_key = os.getenv(provider.env_var)

    if provider.base_url:

        return OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
        )

    return OpenAI(
        api_key=api_key,
    )


# ============================================================
# 4. ACTUAL WEATHER TOOL
# ============================================================

def get_weather(city: str):

    # This is mock data.
    # Replace this function with a real weather API later.

    weather_data = {

        "Hyderabad": {
            "temperature": 30,
            "condition": "Sunny",
        },

        "Mumbai": {
            "temperature": 28,
            "condition": "Cloudy",
        },

        "Delhi": {
            "temperature": 32,
            "condition": "Clear",
        },

        "Bangalore": {
            "temperature": 25,
            "condition": "Rainy",
        },
    }

    return weather_data.get(city)


# ============================================================
# 5. WEATHER TOOL SCHEMA
# ============================================================

weather_tool_schema = {

    "type": "function",

    "function": {

        "name": "get_weather",

        "description": (
            "Get the current weather information "
            "for a city."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "city": {
                    "type": "string",
                    "description": "Name of the city",
                }

            },

            "required": ["city"],

            "additionalProperties": False,
        },
    },
}


# ============================================================
# 6. MAIN LLM FUNCTION
# ============================================================

def llm_reply_with_tool_call(prompt: str) -> str:

    # --------------------------------------------------------
    # Select provider
    # --------------------------------------------------------

    provider = select_provider()

    print(
        f"Using {provider.name} provider"
    )

    # --------------------------------------------------------
    # Create client
    # --------------------------------------------------------

    client = build_client(provider)

    # --------------------------------------------------------
    # Initial messages
    # --------------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use the weather tool when the user "
                "asks about weather. "
                "If the weather tool returns None, "
                "say: I don't have information for this city."
            ),
        },

        {
            "role": "user",
            "content": prompt,
        },
    ]

    # ========================================================
    # FIRST LLM CALL
    # ========================================================

    result = client.chat.completions.create(

        model=provider.model,

        messages=messages,

        max_tokens=200,

        tools=[
            weather_tool_schema
        ],
    )

    # --------------------------------------------------------
    # Get assistant message
    # --------------------------------------------------------

    message = result.choices[0].message

    # ========================================================
    # CASE 1:
    # LLM does NOT need a tool
    # ========================================================

    if not message.tool_calls:

        return message.content


    # ========================================================
    # CASE 2:
    # LLM requested a tool
    # ========================================================

    for tool_call in message.tool_calls:

        # ----------------------------------------------------
        # Get tool name
        # ----------------------------------------------------

        tool_name = tool_call.function.name

        # ----------------------------------------------------
        # Get arguments
        # ----------------------------------------------------

        arguments = json.loads(
            tool_call.function.arguments
        )

        print(
            f"Tool requested: {tool_name}"
        )

        print(
            f"Tool arguments: {arguments}"
        )

        # ----------------------------------------------------
        # Execute tool
        # ----------------------------------------------------

        if tool_name == "get_weather":

            weather_result = get_weather(
                arguments["city"]
            )

        else:

            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        print(
            f"Tool result: {weather_result}"
        )

        # ----------------------------------------------------
        # Add assistant tool-call message
        # ----------------------------------------------------

        messages.append(message)

        # ----------------------------------------------------
        # Add tool result
        # ----------------------------------------------------

        messages.append({

            "role": "tool",

            "tool_call_id": tool_call.id,

            "content": json.dumps(
                weather_result
            ),
        })


    # ========================================================
    # SECOND LLM CALL
    # ========================================================

    result = client.chat.completions.create(

        model=provider.model,

        messages=messages,

        max_tokens=200,

        tools=[
            weather_tool_schema
        ],
    )

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    return result.choices[0].message.content


# ============================================================
# 7. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    user_prompt = input(
        "You: "
    )

    response = llm_reply_with_tool_call(
        user_prompt
    )

    print(
        f"\nAssistant: {response}"
    )