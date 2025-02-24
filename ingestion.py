from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document

load_dotenv()

def read_text_file(file_path: str) -> str:
    """Read a text file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_and_process_documents():
    """
    Load documents from the docs directory, process them, and store in vector database
    """
    print("\n=== Iniciando processamento de documentos ===")
    
    # Create docs directory if it doesn't exist
    if not os.path.exists("docs"):
        os.makedirs("docs")
        print("Pasta 'docs' criada. Adicione seus documentos nela.")
        return None
    
    documents: List[Document] = []
    
    # Load all .txt files from the docs directory
    print("\nArquivos encontrados:")
    for filename in os.listdir("docs"):
        print(f"- Encontrado: {filename}")
        if filename.endswith(".txt"):
            try:
                file_path = os.path.join("docs", filename)
                content = read_text_file(file_path)
                print(f"  Conteúdo ({len(content)} caracteres):")
                print(f"  {content[:100]}...")  # Mostra os primeiros 100 caracteres
                
                doc = Document(
                    page_content=content,
                    metadata={"source": filename}
                )
                documents.append(doc)
                print(f"  ✓ Carregado com sucesso")
            except Exception as e:
                print(f"  ✗ Erro ao carregar {filename}: {str(e)}")
    
    if not documents:
        print("\nNenhum documento .txt encontrado na pasta docs.")
        return None
    
    print(f"\nTotal de documentos carregados: {len(documents)}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=20
    )
    doc_splits = text_splitter.split_documents(documents)
    print(f"Documentos divididos em {len(doc_splits)} chunks")

    # Create and store embeddings
    print("\nCriando vector store...")
    
    # Clear existing collection if it exists
    if os.path.exists("./.chroma"):
        import shutil
        shutil.rmtree("./.chroma")
        print("Vector store anterior removido")
    
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="local-docs",
        embedding=embedding_function,
        persist_directory="./.chroma"
    )
    
    # Test retrieval
    print("\nTestando recuperação...")
    sample_query = doc_splits[0].page_content[:50]  # usa parte do primeiro documento como query
    results = vectorstore.similarity_search(sample_query, k=1)
    if results:
        print("✓ Teste de recuperação bem sucedido")
    else:
        print("✗ Teste de recuperação falhou")
    
    print("\nVector store criado e persistido")
    return vectorstore

# Initialize the retriever
print("\n=== Inicializando retriever ===")
vectorstore = None
if os.path.exists("./.chroma"):
    print("Usando vector store existente")
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name="local-docs",
        persist_directory="./.chroma",
        embedding_function=embedding_function
    )
else:
    print("Criando novo vector store")
    vectorstore = load_and_process_documents()

# Export the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
) if vectorstore else None

print("\n=== Status final ===")
print(f"Retriever inicializado: {retriever is not None}")

if __name__ == "__main__":
    if not vectorstore:
        vectorstore = load_and_process_documents()
        retriever = vectorstore.as_retriever() if vectorstore else None