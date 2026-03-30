"""Quick debug script — run once, then delete."""
from langchain_community.document_loaders import PyMuPDFLoader
from openai import OpenAI
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.testset.transforms.default import default_transforms
from config import OPENAI_API_KEY, JUDGE_MODEL

loader = PyMuPDFLoader("../TEST-FILE/pdf/APPLE_2022_10K.pdf")
docs = loader.load()
print(f"Pages loaded: {len(docs)}")
print(f"First page chars: {len(docs[0].page_content)}")

client = OpenAI(api_key=OPENAI_API_KEY)
llm = llm_factory(JUDGE_MODEL, client=client)
embeddings = RagasOpenAIEmbeddings(model="text-embedding-3-small", client=client)

transforms = default_transforms(documents=docs, llm=llm, embedding_model=embeddings)
print(f"Transforms: {[type(t).__name__ for t in transforms]}")

generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
testset = generator.generate_with_langchain_docs(docs[:20], testset_size=3, transforms=transforms)

rows = testset.to_list()
print(f"\nRows generated: {len(rows)}")
if rows:
    print(f"Keys: {list(rows[0].keys())}")
    for r in rows:
        print(r)
