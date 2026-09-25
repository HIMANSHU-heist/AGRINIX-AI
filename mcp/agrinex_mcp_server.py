"""
AGRINEX AI — MCP Server
==========================
Exposes AGRINEX's data sources (mandi prices, crop calendar knowledge,
soil heuristics) as standard MCP tools, so ANY MCP-compatible agent
(Claude, a LangGraph agent, etc.) can call them through a uniform
protocol instead of hardcoded function calls baked into one app.

Run:
    pip install mcp
    python mcp/agrinex_mcp_server.py

Then point any MCP client (Claude Desktop config, or a custom agent's
MCP client) at this server's stdio transport.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import requests

server = Server("agrinex-ai-tools")

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
MANDI_BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_mandi_price",
            description="Fetch recent mandi (market) prices for a crop/commodity anywhere in India, from the government Agmarknet dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "commodity": {"type": "string", "description": "Crop/commodity name, e.g. 'Tomato', 'Wheat', 'Onion'"},
                    "state": {"type": "string", "description": "Indian state name, optional — omit for all-India results"},
                },
                "required": ["commodity"],
            },
        ),
        Tool(
            name="get_crop_calendar_info",
            description="Retrieve regional crop-calendar knowledge (which crops are traditionally grown in a state/season) from AGRINEX's knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {"type": "string", "description": "Indian state name"},
                    "season": {"type": "string", "description": "Kharif, Rabi, or Zaid/Summer"},
                },
                "required": ["state", "season"],
            },
        ),
        Tool(
            name="analyze_soil_health",
            description="Analyze soil pH, nitrogen and organic carbon values and return plain-language health notes and fertilizer suggestions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ph": {"type": "number"},
                    "nitrogen_kg_ha": {"type": "number"},
                    "organic_carbon_pct": {"type": "number"},
                },
                "required": ["ph", "nitrogen_kg_ha", "organic_carbon_pct"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_mandi_price":
        return await _get_mandi_price(arguments)
    elif name == "get_crop_calendar_info":
        return await _get_crop_calendar_info(arguments)
    elif name == "analyze_soil_health":
        return await _analyze_soil_health(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _get_mandi_price(args: dict):
    api_key = os.environ.get("DATA_GOV_API_KEY")
    if not api_key:
        return [TextContent(type="text", text=json.dumps({"error": "DATA_GOV_API_KEY not configured on server"}))]

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 20,
        "filters[commodity]": args["commodity"],
    }
    if args.get("state"):
        params["filters[state]"] = args["state"]

    try:
        r = requests.get(MANDI_BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        records = r.json().get("records", [])
        summary = [
            {
                "market": rec.get("market"),
                "state": rec.get("state"),
                "date": rec.get("arrival_date"),
                "modal_price": rec.get("modal_price"),
            }
            for rec in records[:10]
        ]
        return [TextContent(type="text", text=json.dumps({"commodity": args["commodity"], "records": summary}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _get_crop_calendar_info(args: dict):
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crop_calendar_kb.json")
    try:
        with open(kb_path) as f:
            docs = json.load(f)
        matches = [
            d for d in docs
            if d.get("state", "").lower() == args["state"].lower()
            and d.get("season", "").lower() == args["season"].lower()
        ]
        if not matches:
            matches = [
                d for d in docs
                if args["state"].lower() in d.get("state", "").lower()
            ]
        return [TextContent(type="text", text=json.dumps({"matches": matches[:5]}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _analyze_soil_health(args: dict):
    ph = args["ph"]
    n = args["nitrogen_kg_ha"]
    oc = args["organic_carbon_pct"]
    notes = []

    if ph < 5.5:
        notes.append("Soil is acidic — consider liming before the next crop cycle.")
    elif ph > 7.5:
        notes.append("Soil is alkaline — gypsum application may help.")
    else:
        notes.append("pH is in a healthy range for most crops.")

    if oc < 0.5:
        notes.append("Organic carbon is low — add compost or green manure.")
    else:
        notes.append("Organic carbon level is adequate.")

    if n < 200:
        notes.append("Nitrogen is on the lower side — a split urea application is advisable.")
    else:
        notes.append("Nitrogen level is sufficient for most cereal crops.")

    return [TextContent(type="text", text=json.dumps({"notes": notes}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
