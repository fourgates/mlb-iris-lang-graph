"""
Minimal MLB player tools used by the agent for batting-average queries.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

import requests  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com"


def _make_api_call(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        response = requests.get(f"{MLB_API_BASE}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("MLB API error for %s: %s", endpoint, e)
        return {"error": str(e)}


def search_player(name: str, only_active: bool = True) -> Dict[str, Any]:
    data = _make_api_call("/api/v1/people/search", params={"names": name})
    if "error" in data:
        return data
    players = []
    for person in data.get("people", []):
        if only_active and not person.get("active", False):
            continue
        players.append({
            "id": person["id"],
            "full_name": person.get("fullName"),
            "team": person.get("currentTeam", {}).get("name", "Free Agent"),
        })
    return {"found": len(players), "players": players, "search_term": name}


def get_player_stats(player_id: int) -> Dict[str, Any]:
    current_year = datetime.now().year
    hydrate = f"stats(group=[hitting],type=[season],season={current_year})"
    data = _make_api_call(f"/api/v1/people/{player_id}", params={"hydrate": hydrate})
    if "error" in data or not data.get("people"):
        return data if "error" in data else {"error": "Player not found"}
    player = data["people"][0]
    result: Dict[str, Any] = {"player_id": player_id, "full_name": player.get("fullName", "Unknown"), "stats": {}}
    for stat_group in player.get("stats", []):
        splits = stat_group.get("splits", [])
        if not splits:
            continue
        stats = splits[0].get("stat", {})
        result["stats"]["hitting_season"] = {
            "avg": stats.get("avg", ".000"),
            "ops": stats.get("ops", ".000"),
            "home_runs": stats.get("homeRuns", 0),
            "rbi": stats.get("rbi", 0),
        }
    return result


__all__ = ["search_player", "get_player_stats"]


