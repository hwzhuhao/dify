import json
import logging
from typing import Any

from pydantic import BaseModel, model_validator
from upstash_vector import Index
from upstash_vector.types import QueryResult

from configs import dify_config
from core.rag.datasource.entity.embedding import Embeddings
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.models.document import Document
from models.dataset import Dataset

logger = logging.getLogger(__name__)


class UpstashVectorConfig(BaseModel):
    url: str
    token: str

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict) -> dict:
        if not values.get("url"):
            raise ValueError("config UPSTASH_VECTOR_REST_URL is required")
        if not values.get("token"):
            raise ValueError("config UPSTASH_VECTOR_REST_TOKEN is required")
        return values


class UpstashVector(BaseVector):
    def __init__(self, collection_name: str, config: UpstashVectorConfig):
        super().__init__(collection_name)
        self.client = Index(url=config.url, token=config.token)

    def get_type(self) -> str:
        return VectorType.UPSTASH

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        if texts:
            self.add_texts(texts, embeddings, **kwargs)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        vectors = [
            (doc.metadata["doc_id"], embedding, doc.metadata, doc.page_content)
            for doc, embedding in zip(documents, embeddings)
        ]
        self.client.upsert(vectors=vectors, namespace=self.collection_name)

    def text_exists(self, id: str) -> bool:
        result = self.client.fetch(ids=id, namespace=self.collection_name)
        return len(result) > 0

    def delete_by_ids(self, ids: list[str]) -> None:
        self.client.delete(ids=ids, namespace=self.collection_name)

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        ids = []
        result = self.client.query(
            filter=f"{key}={value}", namespace=self.collection_name, include_metadata=True, include_data=True
        )
        for res in result:
            ids.append(res.id)
        self.client.delete(ids=ids, namespace=self.collection_name)

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        results: list[QueryResult] = self.client.query(
            vector=query_vector,
            top_k=kwargs.get("top_k", 4),
            include_vectors=True,
            include_metadata=True,
            include_data=True,
        )
        score_threshold = float(kwargs.get("score_threshold") or 0.0)
        docs = []
        for index in range(len(results)):
            distance = results[index].score
            doc_metadata = results[index].metadata
            document = results[index].data
            if distance >= score_threshold:
                doc_metadata["score"] = distance
                doc = Document(
                    page_content=document,
                    metadata=doc_metadata,
                )
                docs.append(doc)
        # Sort the documents by score in descending order
        docs = sorted(docs, key=lambda x: x.metadata["score"], reverse=True)
        return docs

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        # upstash does not support BM25 full text searching
        return []

    def delete(self) -> None:
        self.client.delete_namespace(namespace=self.collection_name)


class UpstashVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> UpstashVector:
        if dataset.index_struct_dict:
            class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
            collection_name = class_prefix
        else:
            dataset_id = dataset.id
            collection_name = Dataset.gen_collection_name_by_id(dataset_id)
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.PGVECTOR, collection_name))

        return UpstashVector(
            collection_name=collection_name,
            config=UpstashVectorConfig(
                url=dify_config.UPSTASH_VECTOR_REST_URL,
                token=dify_config.UPSTASH_VECTOR_REST_TOKEN,
            ),
        )
