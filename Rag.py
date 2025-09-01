import gradio as gr
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline


# 1. Load & Split Documents
def load_and_split_documents(folder_path="pdfs"):
    loader = PyPDFDirectoryLoader(folder_path)
    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return text_splitter.split_documents(docs_before_split)


# 2. HuggingFace Embeddings
def huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        #model_kwargs = {'device' : 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


# 3. Create Vector Store
def create_vector_store(docs_after_split, embeddings):
    vector_store = FAISS.from_documents(docs_after_split, embeddings)
    return vector_store


# 4. Retriever
def retriever(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever


# 5. LLM Model (Falcon 7B)

def llm():

  local_llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",

        model_kwargs={
            "temperature": 0.2,
            "do_sample": True
        },
        pipeline_kwargs={
            "max_new_tokens": 128
        }
    )

  return local_llm

# 6. Prompt Template
def prompt_template():
    prompt_template = """You are an expert assistant. 
Use the following context to answer the user's question. 

Guidelines:
- If the context contains the answer, provide it concisely in no more than five sentences. 
- If the context does not contain the answer, reply: "I can't find the final answer but you may want to check the following links".

Do not repeat the guidelines. Only output the answer.

Context:
{context}

Question: {question}

Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return PROMPT




# 7. Retrieval QA
def retrieval_qa(llm, retriever, PROMPT):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return retrieval_qa


# === Pipeline Setup ===
docs_after_split = load_and_split_documents()
embeddings = huggingface_embeddings()
vector_store = create_vector_store(docs_after_split, embeddings)
retr = retriever(vector_store)
model = llm()
prompt = prompt_template()
qa_chain = retrieval_qa(model, retr, prompt)


def chat_with_model(history, new_message):
    # Run the QA chain
    
    result = qa_chain({"query": new_message})  

    # Get retrieved documents
    relevant_docs = result['source_documents']
    
    # Format answer with sources
    answer_text = result['result'] + "\n\n" + ("*"*50) + "\nRetrieved Sources:\n"
    for i, doc in enumerate(relevant_docs):
        answer_text += f"{i+1}. {doc.metadata.get('source', 'Unknown')}"

    history.append((new_message, answer_text))
    return history, ""



# 9. Gradio App
def gradio_chat_app():
    with gr.Blocks() as app:
        gr.Markdown("# AI_course Model Chat Interface")
        gr.Markdown("Chat with the model in a conversational format.")
        
        chatbot = gr.Chatbot(label="Chat Interface")
        user_input = gr.Textbox(label="Your message", placeholder="Type something...", lines=1)
        send_button = gr.Button("Send")
        
        def clear_chat():
            return [], ""
    
        clear_button = gr.Button("Clear Chat")
        
        send_button.click(
            fn=chat_with_model,
            inputs=[chatbot, user_input],
            outputs=[chatbot, user_input]
        )
        clear_button.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, user_input]
        )
        
    return app


# 10. Launch App
if __name__ == "__main__":
    app = gradio_chat_app()
    app.launch(share=True)
