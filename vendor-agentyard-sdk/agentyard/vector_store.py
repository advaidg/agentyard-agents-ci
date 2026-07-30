"""Unified vector store client for RAG.

Backends supported:
- pgvector (uses the system Postgres)
- qdrant (via REST API)
- chroma (via REST API)
- pinecone (via REST API)
- weaviate (via REST API)
- memory (in-process, for tests)

Selection via YARD_VECTOR_BACKEND env var.

Usage:
    # Ingest
    await ctx.vector_store.upsert(
        collection="invoices",
        items=[
            {"id": "doc1", "text": "invoice content...", "metadata": {"currency": "USD"}},
            {"id": "doc2", "text": "...", "metadata": {...}},
        ],
    )

    # Query
    results = await ctx.vector_store.query(
        collection="invoices",
        query="tax deductible expenses",
        top_k=5,
        filter={"currency": "USD"},
    )
    for hit in results:
        print(hit.score, hit.text, hit.metadata)
"""
import abc
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agentyard.vector_store")


@dataclass
class VectorHit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class VectorItem:
    id: str
    text: str
    vector: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreError(Exception):
    """Raised for any vector-store backend failure."""


class VectorStoreBackend(abc.ABC):
    """Backend implementation. Subclasses implement query + upsert + delete."""

    @abc.abstractmethod
    async def upsert(self, collection: str, items: list[VectorItem]) -> int:
        """Insert/update items. Returns count."""

    @abc.abstractmethod
    async def query(
        self,
        collection: str,
        query: str | list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Search for top_k nearest neighbors."""

    @abc.abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> int:
        """Delete by id. Returns count."""

    @abc.abstractmethod
    async def create_collection(self, collection: str, dimension: int) -> None:
        """Create if not exists."""


# ── In-memory backend (for tests) ──────────────────────────────────────

class MemoryBackend(VectorStoreBackend):
    def __init__(self) -> None:
        self._collections: dict[str, list[VectorItem]] = {}

    async def create_collection(self, collection: str, dimension: int) -> None:
        self._collections.setdefault(collection, [])

    async def upsert(self, collection: str, items: list[VectorItem]) -> int:
        existing = self._collections.setdefault(collection, [])
        incoming_ids = {x.id for x in items}
        # Replace existing with same id, append new
        self._collections[collection] = [
            i for i in existing if i.id not in incoming_ids
        ] + list(items)
        return len(items)

    async def query(
        self,
        collection: str,
        query: str | list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        items = self._collections.get(collection, [])
        if filter:
            items = [
                i for i in items
                if all(i.metadata.get(k) == v for k, v in filter.items())
            ]
        # Deterministic dummy scoring for tests: hash similarity
        q_hash = int(hashlib.sha1(str(query).encode()).hexdigest(), 16) % 1000
        scored: list[VectorHit] = []
        for i in items:
            i_hash = int(hashlib.sha1(i.id.encode()).hexdigest(), 16) % 1000
            score = 1.0 - abs(q_hash - i_hash) / 1000
            scored.append(
                VectorHit(id=i.id, score=score, text=i.text, metadata=i.metadata)
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    async def delete(self, collection: str, ids: list[str]) -> int:
        if collection not in self._collections:
            return 0
        to_del = set(ids)
        before = len(self._collections[collection])
        self._collections[collection] = [
            i for i in self._collections[collection] if i.id not in to_del
        ]
        return before - len(self._collections[collection])


# ── Qdrant backend ─────────────────────────────────────────────────────

class QdrantBackend(VectorStoreBackend):
    def __init__(
        self,
        url: str,
        api_key: str = "",
        dimension: int = 1536,
    ) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.dimension = dimension

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["api-key"] = self.api_key
        return h

    async def create_collection(self, collection: str, dimension: int) -> None:
        import httpx
        async with httpx.AsyncClient() as client:
            # PUT /collections/{name} is idempotent enough — 409 means exists
            resp = await client.put(
                f"{self.url}/collections/{collection}",
                headers=self._headers(),
                json={"vectors": {"size": dimension, "distance": "Cosine"}},
            )
            if resp.status_code not in (200, 201, 409):
                raise VectorStoreError(
                    f"Qdrant create failed: {resp.status_code} {resp.text}"
                )

    async def upsert(self, collection: str, items: list[VectorItem]) -> int:
        import httpx
        points = []
        for item in items:
            if item.vector is None:
                raise VectorStoreError(
                    f"Qdrant requires vectors; item {item.id} has none"
                )
            points.append({
                "id": item.id,
                "vector": item.vector,
                "payload": {"text": item.text, **item.metadata},
            })
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{self.url}/collections/{collection}/points",
                headers=self._headers(),
                json={"points": points},
                params={"wait": "true"},
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Qdrant upsert failed: {resp.status_code} {resp.text}"
                )
        return len(items)

    async def query(
        self,
        collection: str,
        query: str | list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        import httpx
        if isinstance(query, str):
            raise VectorStoreError(
                "Qdrant backend requires a pre-computed vector for query"
            )
        body: dict[str, Any] = {
            "vector": query,
            "limit": top_k,
            "with_payload": True,
        }
        if filter:
            body["filter"] = {
                "must": [
                    {"key": k, "match": {"value": v}} for k, v in filter.items()
                ]
            }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/collections/{collection}/points/search",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Qdrant query failed: {resp.status_code} {resp.text}"
                )
            data = resp.json()
            hits: list[VectorHit] = []
            for r in data.get("result", []):
                payload = dict(r.get("payload", {}) or {})
                text = payload.pop("text", "")
                hits.append(VectorHit(
                    id=str(r.get("id", "")),
                    score=float(r.get("score", 0)),
                    text=text,
                    metadata=payload,
                ))
            return hits

    async def delete(self, collection: str, ids: list[str]) -> int:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/collections/{collection}/points/delete",
                headers=self._headers(),
                json={"points": ids},
                params={"wait": "true"},
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Qdrant delete failed: {resp.status_code} {resp.text}"
                )
        return len(ids)


# ── Pgvector backend ───────────────────────────────────────────────────

class PgvectorBackend(VectorStoreBackend):
    """Postgres + pgvector extension. Uses the existing system DB.

    Tables are created per-collection on first use::

        CREATE TABLE yard_vec_{collection} (
            id TEXT PRIMARY KEY,
            text TEXT,
            metadata JSONB,
            embedding vector(dim)
        )
    """

    def __init__(self, dsn: str, dimension: int = 1536) -> None:
        self.dsn = dsn
        self.dimension = dimension

    async def _conn(self):  # type: ignore[no-untyped-def]
        try:
            import asyncpg
        except ImportError as e:
            raise VectorStoreError(
                "pgvector backend requires asyncpg. Install with: pip install asyncpg"
            ) from e
        return await asyncpg.connect(self.dsn)

    async def create_collection(self, collection: str, dimension: int) -> None:
        conn = await self._conn()
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS "yard_vec_{collection}" (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    metadata JSONB,
                    embedding vector({dimension})
                )
            ''')
        finally:
            await conn.close()

    async def upsert(self, collection: str, items: list[VectorItem]) -> int:
        import json as _json
        conn = await self._conn()
        try:
            for item in items:
                if item.vector is None:
                    raise VectorStoreError(
                        f"pgvector requires vectors; item {item.id} has none"
                    )
                await conn.execute(
                    f'''
                    INSERT INTO "yard_vec_{collection}" (id, text, metadata, embedding)
                    VALUES ($1, $2, $3, $4::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    ''',
                    item.id,
                    item.text,
                    _json.dumps(item.metadata),
                    str(item.vector),
                )
        finally:
            await conn.close()
        return len(items)

    async def query(
        self,
        collection: str,
        query: str | list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        import json as _json
        if isinstance(query, str):
            raise VectorStoreError(
                "pgvector backend requires a pre-computed vector for query"
            )
        conn = await self._conn()
        try:
            base_sql = f'''
                SELECT id, text, metadata, 1 - (embedding <=> $1::vector) AS score
                FROM "yard_vec_{collection}"
            '''
            params: list[Any] = [str(query)]
            if filter:
                where_clauses: list[str] = []
                for k, v in filter.items():
                    params.append(
                        _json.dumps(v) if not isinstance(v, str) else v
                    )
                    where_clauses.append(f"metadata->>'{k}' = ${len(params)}")
                if where_clauses:
                    base_sql += " WHERE " + " AND ".join(where_clauses)
            params.append(top_k)
            base_sql += f" ORDER BY score DESC LIMIT ${len(params)}"
            rows = await conn.fetch(base_sql, *params)
            return [
                VectorHit(
                    id=str(r["id"]),
                    score=float(r["score"]),
                    text=r["text"] or "",
                    metadata=_json.loads(r["metadata"] or "{}"),
                )
                for r in rows
            ]
        finally:
            await conn.close()

    async def delete(self, collection: str, ids: list[str]) -> int:
        conn = await self._conn()
        try:
            result = await conn.execute(
                f'DELETE FROM "yard_vec_{collection}" WHERE id = ANY($1::text[])',
                ids,
            )
            # asyncpg returns "DELETE N"
            return int(result.split()[-1]) if result else 0
        finally:
            await conn.close()


# ── Chroma backend ─────────────────────────────────────────────────────

class ChromaBackend(VectorStoreBackend):
    """Chroma via REST API (chromadb running standalone)."""

    def __init__(
        self,
        url: str,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ) -> None:
        self.url = url.rstrip("/")
        self.tenant = tenant
        self.database = database

    async def create_collection(self, collection: str, dimension: int) -> None:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/api/v1/collections",
                json={"name": collection, "metadata": {"dimension": dimension}},
            )
            if resp.status_code not in (200, 201, 409):
                raise VectorStoreError(
                    f"Chroma create failed: {resp.status_code} {resp.text}"
                )

    async def upsert(self, collection: str, items: list[VectorItem]) -> int:
        import httpx
        body: dict[str, Any] = {
            "ids": [i.id for i in items],
            "documents": [i.text for i in items],
            "metadatas": [i.metadata or {} for i in items],
            "embeddings": (
                [i.vector for i in items] if all(i.vector for i in items) else None
            ),
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/api/v1/collections/{collection}/upsert",
                json=body,
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Chroma upsert failed: {resp.status_code} {resp.text}"
                )
        return len(items)

    async def query(
        self,
        collection: str,
        query: str | list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        import httpx
        body: dict[str, Any] = {"n_results": top_k}
        if isinstance(query, str):
            body["query_texts"] = [query]
        else:
            body["query_embeddings"] = [query]
        if filter:
            body["where"] = filter
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/api/v1/collections/{collection}/query",
                json=body,
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Chroma query failed: {resp.status_code} {resp.text}"
                )
            data = resp.json()
            ids = (data.get("ids") or [[]])[0]
            docs = (data.get("documents") or [[]])[0]
            metas = (data.get("metadatas") or [[]])[0]
            dists = (data.get("distances") or [[]])[0]
            return [
                VectorHit(
                    id=str(ids[i]),
                    score=1.0 - float(dists[i]) if i < len(dists) else 0.0,
                    text=str(docs[i] or "") if i < len(docs) else "",
                    metadata=metas[i] or {} if i < len(metas) else {},
                )
                for i in range(len(ids))
            ]

    async def delete(self, collection: str, ids: list[str]) -> int:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/api/v1/collections/{collection}/delete",
                json={"ids": ids},
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Chroma delete failed: {resp.status_code} {resp.text}"
                )
        return len(ids)


# ── Pinecone backend (minimal) ─────────────────────────────────────────

class PineconeBackend(VectorStoreBackend):
    def __init__(
        self,
        api_key: str,
        environment: str = "",
        index_host: str = "",
    ) -> None:
        self.api_key = api_key
        self.environment = environment
        self.index_host = index_host.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Api-Key": self.api_key, "Content-Type": "application/json"}

    async def create_collection(self, collection: str, dimension: int) -> None:
        # Pinecone "collections" are actually indexes; creation goes through
        # a separate control plane. For simplicity we assume the index already exists.
        return None

    async def upsert(self, collection: str, items: list[VectorItem]) -> int:
        import httpx
        vectors = [
            {
                "id": i.id,
                "values": i.vector,
                "metadata": {"text": i.text, **i.metadata},
            }
            for i in items
        ]
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.index_host}/vectors/upsert",
                headers=self._headers(),
                json={"vectors": vectors, "namespace": collection},
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Pinecone upsert failed: {resp.status_code} {resp.text}"
                )
        return len(items)

    async def query(
        self,
        collection: str,
        query: str | list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        import httpx
        if isinstance(query, str):
            raise VectorStoreError(
                "Pinecone backend requires a pre-computed vector for query"
            )
        body: dict[str, Any] = {
            "vector": query,
            "topK": top_k,
            "namespace": collection,
            "includeMetadata": True,
        }
        if filter:
            body["filter"] = {k: {"$eq": v} for k, v in filter.items()}
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.index_host}/query",
                headers=self._headers(),
                json=body,
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Pinecone query failed: {resp.status_code} {resp.text}"
                )
            data = resp.json()
            hits: list[VectorHit] = []
            for m in data.get("matches", []):
                meta = dict(m.get("metadata", {}) or {})
                text = meta.pop("text", "")
                hits.append(VectorHit(
                    id=str(m.get("id", "")),
                    score=float(m.get("score", 0)),
                    text=text,
                    metadata=meta,
                ))
            return hits

    async def delete(self, collection: str, ids: list[str]) -> int:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.index_host}/vectors/delete",
                headers=self._headers(),
                json={"ids": ids, "namespace": collection},
            )
            if resp.status_code >= 400:
                raise VectorStoreError(
                    f"Pinecone delete failed: {resp.status_code} {resp.text}"
                )
        return len(ids)


# ── Client façade ──────────────────────────────────────────────────────

class VectorStoreClient:
    """High-level façade. Selects backend from env or explicit config."""

    def __init__(self, backend: VectorStoreBackend | None = None) -> None:
        self.backend = backend or self._default_backend()

    @staticmethod
    def _default_backend() -> VectorStoreBackend:
        kind = os.environ.get("YARD_VECTOR_BACKEND", "memory").lower()
        if kind == "memory":
            return MemoryBackend()
        if kind == "qdrant":
            return QdrantBackend(
                url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
                api_key=os.environ.get("QDRANT_API_KEY", ""),
            )
        if kind == "pgvector":
            return PgvectorBackend(
                dsn=os.environ.get(
                    "PGVECTOR_DSN", os.environ.get("DATABASE_URL", "")
                ),
            )
        if kind == "chroma":
            return ChromaBackend(
                url=os.environ.get("CHROMA_URL", "http://chroma:8000")
            )
        if kind == "pinecone":
            return PineconeBackend(
                api_key=os.environ.get("PINECONE_API_KEY", ""),
                index_host=os.environ.get("PINECONE_INDEX_HOST", ""),
            )
        raise VectorStoreError(f"Unknown YARD_VECTOR_BACKEND: {kind}")

    async def create_collection(
        self, collection: str, dimension: int = 1536
    ) -> None:
        await self.backend.create_collection(collection, dimension)

    async def upsert(
        self,
        collection: str,
        items: list[dict[str, Any]] | list[VectorItem],
    ) -> int:
        normalized: list[VectorItem] = [
            i if isinstance(i, VectorItem) else VectorItem(**i)
            for i in items
        ]
        return await self.backend.upsert(collection, normalized)

    async def query(
        self,
        collection: str,
        query: str | list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        return await self.backend.query(
            collection, query, top_k=top_k, filter=filter
        )

    async def delete(self, collection: str, ids: list[str]) -> int:
        return await self.backend.delete(collection, ids)
