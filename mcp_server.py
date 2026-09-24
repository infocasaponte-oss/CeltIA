from mcp.server.fastmcp import FastMCP

from core.policy import ToolPolicy
from core.tools import builtins, calculator, python_exec, read_file

mcp = FastMCP("celtia-tools", json_response=True)
policy = ToolPolicy()
registry = builtins(policy=policy)


@mcp.tool()
def calculate(expression: str) -> dict:
    """Evaluate safe arithmetic."""
    policy.validate("calculator", {"expression": expression})
    return calculator(expression)


@mcp.tool()
def execute_python(code: str, timeout: int = 8) -> dict:
    """Run Python in a restricted subprocess."""
    return python_exec(code, timeout)


@mcp.tool()
def read_workspace_file(path: str) -> dict:
    """Read a text file under /workspace."""
    return read_file(path)


@mcp.tool()
def tool_registry() -> dict:
    """Expose the allowed tool names for policy inspection."""
    return {"tools": sorted(registry.tools.keys())}


if __name__ == "__main__":
    mcp.run()
