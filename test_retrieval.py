from dotenv import load_dotenv
import os
from ingestion import retriever
from pprint import pprint

load_dotenv()

def test_retrieval():
    # Teste básico do retriever
    print("Testando retriever...")
    
    # Verifica se o retriever existe
    if not retriever:
        print("Erro: Retriever não foi inicializado!")
        return
        
    # Faz uma busca de teste
    docs = retriever.invoke("test query")
    
    print("\nDocumentos recuperados:")
    for doc in docs:
        print("\n---Documento---")
        print(f"Conteúdo: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        
    print("\nTotal de documentos:", len(docs))
    
    # Mostra conteúdo do diretório docs
    print("\nArquivos na pasta docs:")
    if os.path.exists("docs"):
        for file in os.listdir("docs"):
            print(f"- {file}")
    else:
        print("Pasta docs não encontrada!")
        
    # Mostra se o diretório .chroma existe
    print("\nStatus do .chroma:")
    if os.path.exists(".chroma"):
        print("Diretório .chroma existe")
    else:
        print("Diretório .chroma não existe!")

if __name__ == "__main__":
    test_retrieval()