"""AgentYard API client — HTTP client for the registry service."""

import httpx


class AgentYardClient:
    """Client for interacting with an AgentYard registry."""

    def __init__(self, registry_url: str = "http://localhost:8000", token: str = ""):
        self.registry_url = registry_url.rstrip("/")
        self.token = token
        self._headers = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _url(self, path: str) -> str:
        return f"{self.registry_url}/api{path}"

    def _check(self, resp: httpx.Response) -> dict:
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise AgentYardError(data["error"]["message"])
        return data.get("data")

    # ── Agents ──

    def register_agent(self, payload: dict) -> dict:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(self._url("/agents"), json=payload, headers=self._headers)
            return self._check(resp)

    def list_agents(
        self,
        namespace: str | None = None,
        framework: str | None = None,
        q: str | None = None,
        limit: int = 50,
    ) -> dict:
        params = {"limit": limit}
        if namespace:
            params["namespace"] = namespace
        if framework:
            params["framework"] = framework
        if q:
            params["q"] = q

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(self._url("/agents"), params=params, headers=self._headers)
            return self._check(resp)

    def get_agent(self, agent_id: str) -> dict:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(self._url(f"/agents/{agent_id}"), headers=self._headers)
            return self._check(resp)

    def search_agents(self, query: str) -> dict:
        return self.list_agents(q=query)

    def deprecate_agent(self, agent_id: str, note: str) -> dict:
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(
                self._url(f"/agents/{agent_id}"),
                params={"deprecation_note": note},
                headers=self._headers,
            )
            return self._check(resp)

    def check_health(self, agent_id: str) -> dict:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(self._url(f"/agents/{agent_id}/health"), headers=self._headers)
            return self._check(resp)

    # ── Platform ──

    def platform_health(self) -> dict:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(self._url("/health"), headers=self._headers)
            return self._check(resp)

    def platform_stats(self) -> dict:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(self._url("/stats"), headers=self._headers)
            return self._check(resp)

    # ── Systems ──

    def list_systems(self, namespace: str | None = None, limit: int = 50) -> dict:
        params: dict = {"limit": limit}
        if namespace:
            params["namespace"] = namespace
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                self._url("/systems"), params=params, headers=self._headers
            )
            return self._check(resp)

    def find_system_by_slug_or_name(self, identifier: str) -> dict | None:
        """Look up a system by slug first, falling back to a list+filter by name.

        CLI users call scenarios with `--system invoice-pipeline`, so we need a
        resolver that accepts either a slug, name, or raw UUID.
        """
        # Try slug-style lookup via the /systems/by-slug endpoint.
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                self._url(f"/systems/by-slug?slug={identifier}"),
                headers=self._headers,
            )
            if resp.status_code == 200:
                data = resp.json().get("data")
                if data:
                    return data
        # Fall back to list + name match.
        data = self.list_systems(limit=200)
        items = data.get("items", []) if isinstance(data, dict) else []
        for item in items:
            if (
                item.get("slug") == identifier
                or item.get("name") == identifier
                or item.get("id") == identifier
            ):
                return item
        return None

    # ── Scenarios (G10) ──

    def list_scenarios(self, system_id: str) -> dict:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                self._url(f"/systems/{system_id}/scenarios"),
                headers=self._headers,
            )
            return self._check(resp)

    def run_scenario(self, system_id: str, scenario_id: str) -> dict:
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(
                self._url(f"/systems/{system_id}/scenarios/{scenario_id}/run"),
                json={"triggered_by": "ci"},
                headers=self._headers,
            )
            return self._check(resp)


class AgentYardError(Exception):
    """Raised when the AgentYard API returns an error."""
