from duckduckgo_search import DDGS
import json

def ddg_web_search(query: str, max_results: int = 5):
    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append({"title": r.get("title"), "href": r.get("href"), "snippet": r.get("body")})
    return out

class MCPAdapter:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def call_tool(self, tool_name: str, **kwargs):
        if not self.enabled:
            return f"[MCP disabled] Would call '{tool_name}' with args: {json.dumps(kwargs)[:400]}"
        # replace with real MCP client call when you have one
        return "[MCP enabled placeholder response]"
