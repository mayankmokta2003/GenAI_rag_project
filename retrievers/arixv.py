from langchain_classic.retrievers import ArxivRetriever


retriever = ArxivRetriever(
    load_max_docs = 2,
    load_all_available_meta = True
)

result = retriever.invoke("adagrad optimizer")

for i in result:
    print(i)
    print()