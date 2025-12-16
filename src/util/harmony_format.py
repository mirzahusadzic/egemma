"""
Harmony format utilities for GPT-OSS models.

Implements the official OpenAI Harmony protocol for formatting prompts.
Based on: https://cookbook.openai.com/articles/openai-harmony
"""

from datetime import datetime
from typing import Any


def format_system_message(
    reasoning_effort: str = "high",
    knowledge_cutoff: str = "2024-06",
    current_date: str | None = None,
    has_functions: bool = False,
) -> str:
    """
    Format Harmony system message.

    Args:
        reasoning_effort: Reasoning level ("low", "medium", "high")
        knowledge_cutoff: Model knowledge cutoff date
        current_date: Current date (defaults to today)
        has_functions: Whether function tools are defined

    Returns:
        Formatted system message with Harmony tokens

    Example:
        >>> format_system_message(has_functions=True)
        '<|start|>system<|message|>You are ChatGPT...'
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    content_parts = [
        "You are ChatGPT, a large language model trained by OpenAI.",
        f"Knowledge cutoff: {knowledge_cutoff}",
        f"Current date: {current_date}",
        f"Reasoning: {reasoning_effort}",
        "# Valid channels: analysis, commentary, final. Channel must be "
        "included for every message.",
    ]

    if has_functions:
        content_parts.append(
            "Calls to these tools must go to the commentary channel: 'functions'."
        )

    # Explicitly disable built-in tools (browser, python, git) to prevent
    # model from trying to use them
    content_parts.append(
        "# Note: Built-in tools (browser.search, browser.open, "
        "browser.find, python, git) are not available in this environment."
    )

    # Clarify that bash should be used for git commands
    if has_functions:
        content_parts.append(
            "# Note: Use the 'bash' tool with git commands "
            "(e.g., bash with command='git status')."
        )

    content = "\n".join(content_parts)
    return f"<|start|>system<|message|>{content}<|end|>"


def format_developer_message(
    instructions: str | None = None, tools: list[dict] | None = None
) -> str:
    """
    Format Harmony developer message with instructions and tool definitions.

    Args:
        instructions: System instructions (what's normally the "system prompt")
        tools: List of tool definitions in OpenAI function format

    Returns:
        Formatted developer message with Harmony tokens

    Example:
        >>> format_developer_message(
        ...     instructions="Be helpful",
        ...     tools=[{"type": "function", "function": {...}}]
        ... )
        '<|start|>developer<|message|># Instructions...'
    """
    content_parts = []

    # Add core operational guidelines (always included to prevent common errors)
    core_guidelines = """# Instructions
## Tool Usage
- **CRITICAL: Use 'bash' tool for ALL shell commands** (git, grep, ls, find, etc.)
  - Correct: bash(command='git status')
  - WRONG: Never call git/python/browser as separate tools
- Use glob/grep BEFORE read_file to locate specific content efficiently
- Use limit/offset parameters when reading large files

## Channel Separation (CRITICAL)
- Each message must use ONLY ONE channel (analysis, commentary, OR final)
- NEVER mix channels in a single message
- To call a tool after generating text:
  1. End your message with the appropriate channel
  2. Start a NEW message for the tool call in commentary channel
- Example: Do NOT output "text here assistantcommentary to=functions.tool"
  - This is WRONG - it mixes final text with commentary tool call

## Response Guidelines
- Be concise and helpful
- After gathering context with tools, provide a FINAL answer to the user
- Avoid analysis loops - gather data, decide, then respond in the final channel"""

    content_parts.append(core_guidelines)

    # Add custom instructions if provided
    if instructions:
        content_parts.append(f"\n## Additional Instructions\n{instructions}")

    # Add tools section
    if tools:
        tool_definitions = _format_tools_as_typescript(tools)
        content_parts.append(f"# Tools\n## functions\n{tool_definitions}")

    content = "\n".join(content_parts)
    return f"<|start|>developer<|message|>{content}<|end|>"


def format_user_message(content: str) -> str:
    """
    Format user message with Harmony tokens.

    Args:
        content: Message content

    Returns:
        Formatted user message

    Example:
        >>> format_user_message("Hello!")
        '<|start|>user<|message|>Hello!<|end|>'
    """
    return f"<|start|>user<|message|>{content}<|end|>"


def format_assistant_message(content: str, channel: str = "final") -> str:
    """
    Format assistant message with Harmony tokens.

    Args:
        content: Message content
        channel: Channel name ("analysis", "commentary", "final")

    Returns:
        Formatted assistant message

    Example:
        >>> format_assistant_message("Hello!", channel="final")
        '<|start|>assistant<|channel|>final<|message|>Hello!<|end|>'
    """
    return f"<|start|>assistant<|channel|>{channel}<|message|>{content}<|end|>"


def format_tool_message(tool_name: str, content: str) -> str:
    """
    Format tool response message with Harmony tokens.

    Args:
        tool_name: Name of the tool (e.g., "functions.get_weather")
        content: Tool output

    Returns:
        Formatted tool message

    Example:
        >>> format_tool_message("functions.get_weather", '{"temp": 20}')
        '<|start|>functions.get_weather to=assistant<|channel|>commentary...'
    """
    return (
        f"<|start|>{tool_name} to=assistant<|channel|>commentary"
        f"<|message|>{content}<|end|>"
    )


def _format_tools_as_typescript(tools: list[dict]) -> str:
    """
    Convert OpenAI function tool definitions to TypeScript-like syntax.

    Args:
        tools: List of tool definitions in OpenAI format

    Returns:
        TypeScript-like tool definitions wrapped in namespace

    Example:
        >>> tools = [{
        ...     "type": "function",
        ...     "function": {
        ...         "name": "get_weather",
        ...         "description": "Gets weather",
        ...         "parameters": {
        ...             "type": "object",
        ...             "properties": {
        ...                 "location": {"type": "string", "description": "City"}
        ...             },
        ...             "required": ["location"]
        ...         }
        ...     }
        ... }]
        >>> print(_format_tools_as_typescript(tools))
        namespace functions {
        // Gets weather
        type get_weather = (_: {
          // City
          location: string,
        }) => any;
        } // namespace functions
    """
    function_defs = []

    for tool in tools:
        if tool.get("type") != "function":
            continue

        func = tool["function"]
        name = func["name"]
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        # Add description as comment
        if description:
            function_defs.append(f"// {description}")

        # Format function signature
        if not parameters or not parameters.get("properties"):
            # No parameters
            function_defs.append(f"type {name} = () => any;\n")
        else:
            # Has parameters
            function_defs.append(f"type {name} = (_: {{")

            # Format each parameter
            props = parameters.get("properties", {})
            required = parameters.get("required", [])

            for prop_name, prop_spec in props.items():
                # Add parameter description as comment
                prop_desc = prop_spec.get("description", "")
                if prop_desc:
                    function_defs.append(f"  // {prop_desc}")

                # Format parameter type
                ts_type = _json_schema_to_typescript_type(prop_spec)
                optional = "?" if prop_name not in required else ""
                default = prop_spec.get("default")
                default_comment = f" // default: {default}" if default else ""

                function_defs.append(
                    f"  {prop_name}{optional}: {ts_type},{default_comment}"
                )

            function_defs.append("}) => any;\n")

    # Wrap in namespace
    tool_code = "\n".join(function_defs)
    return f"namespace functions {{\n{tool_code}}} // namespace functions"


def _json_schema_to_typescript_type(schema: dict[str, Any]) -> str:
    """
    Convert JSON Schema type to TypeScript type.

    Args:
        schema: JSON Schema property specification

    Returns:
        TypeScript type string

    Example:
        >>> _json_schema_to_typescript_type({"type": "string"})
        'string'
        >>> _json_schema_to_typescript_type(
        ...     {"type": "array", "items": {"type": "string"}}
        ... )
        'string[]'
        >>> _json_schema_to_typescript_type({"enum": ["a", "b"]})
        '"a" | "b"'
    """
    # Handle enum
    if "enum" in schema:
        enum_values = [f'"{v}"' for v in schema["enum"]]
        return " | ".join(enum_values)

    # Handle array
    if schema.get("type") == "array":
        items = schema.get("items", {})
        item_type = _json_schema_to_typescript_type(items)
        return f"{item_type}[]"

    # Handle object
    if schema.get("type") == "object":
        return "object"

    # Basic type mapping
    type_map = {
        "string": "string",
        "number": "number",
        "integer": "number",
        "boolean": "boolean",
        "null": "null",
    }

    return type_map.get(schema.get("type", ""), "any")


def format_conversation(
    messages: list[dict],
    instructions: str | None = None,
    tools: list[dict] | None = None,
    reasoning_effort: str = "high",
) -> str:
    """
    Format a complete conversation in Harmony format.

    Args:
        messages: List of message dicts with 'role' and 'content'
        instructions: System instructions (developer message)
        tools: Tool definitions
        reasoning_effort: Reasoning level

    Returns:
        Complete formatted conversation ready for the model

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "What is 2+2?"}
        ... ]
        >>> prompt = format_conversation(messages, instructions="Be helpful")
        >>> "<|start|>system" in prompt
        True
    """
    formatted_parts = []

    # Add system message
    has_functions = bool(tools)
    formatted_parts.append(
        format_system_message(
            reasoning_effort=reasoning_effort,
            has_functions=has_functions,
        )
    )

    # Add developer message if we have instructions or tools
    if instructions or tools:
        formatted_parts.append(format_developer_message(instructions, tools))

    # Format conversation messages
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            # Skip - we already added proper system message
            continue
        elif role == "user":
            formatted_parts.append(format_user_message(content))
        elif role == "assistant":
            # Check if this message has tool calls
            tool_calls = msg.get("tool_calls", [])

            if tool_calls:
                # Format tool calls in Harmony protocol format
                # Per cookbook: line 303: <|channel|>commentary
                # to=functions.NAME <|constrain|>json<|message|>{args}<|call|>
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    arguments = tool_call.get("arguments", "{}")

                    # Harmony tool call format
                    formatted_parts.append(
                        f"<|start|>assistant<|channel|>commentary "
                        f"to=functions.{tool_name} "
                        f"<|constrain|>json<|message|>{arguments}<|call|>"
                    )
            else:
                # Regular assistant message (no tool calls)
                channel = msg.get("channel", "final")
                formatted_parts.append(format_assistant_message(content, channel))
        elif role == "tool" or role.startswith("functions."):
            # Tool response
            tool_name = msg.get("name", role)
            formatted_parts.append(format_tool_message(tool_name, content))

    # Add incomplete assistant message to prompt completion
    formatted_parts.append("<|start|>assistant")

    return "".join(formatted_parts)
