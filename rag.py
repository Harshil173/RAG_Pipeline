from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Configuring LLM Through HuggingFace (using open source models for embedding as well as text-generation)
textGenLLM = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)
model = ChatHuggingFace(llm=textGenLLM)

textEmbedLLM = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load the Document
loader = PyPDFLoader('factory_spec.pdf')
docs = loader.load()

# Splitt the Document into Text Chunks
splitter =  RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 150,
)
chunks = splitter.split_documents(docs) 

# Converting these chunks into vector embeddings
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=textEmbedLLM
)

# Creating an Retriever  
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":2})

# Preparing the Prompt Template (Augmentation Part)
prompt =  PromptTemplate(
    template = """
    You are a helpful assistant.
    Answer only from the provided document context.
    If the context is insufficient, just say don't know.
    {context}
    Question : {question}
""",
    input_variables=['context','question']
)


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke('What is the maximum allowed positional accuracy for Arm C, and what is its maximum joint temperature?')

print(result)
