import tkinter as tk
from tkinter import scrolledtext
from torch import cuda, bfloat16
import transformers
import re
import os

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


def trim_answer(full_answer):
    helpful_answer_index = full_answer.find("Helpful Answer")
    if helpful_answer_index != -1:
        helpful_answer = full_answer[helpful_answer_index + len("Helpful Answer: "):]
        return helpful_answer
    else:
        return "Cannot find a suitable answer."

def create_llm(model_id, hf_auth, device):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth
    )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth
    )
    
    model.eval()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=hf_auth,
        padding_side="right"
    )
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        eos_token_id=terminators,
        temperature=0.1,
        max_new_tokens=1024,
        repetition_penalty=1.1,
    )
    
    llm = HuggingFacePipeline(pipeline=generate_text)
    
    return llm

def create_retriever(llm, model_name, model_kwargs):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    vectorstore = FAISS.load_local("NewPhase2FaissIndex", embeddings, allow_dangerous_deserialization=True)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), return_source_documents=True)
    return chain

def generate_answer(query, chain, chat_history):
    result = chain({
        "question": query,
        "chat_history": chat_history,
    })
    
    helpful_answer = trim_answer(result['answer'])
    source_docs = result['source_documents']
    
    chat_history.append((query, helpful_answer))
    
    return helpful_answer, source_docs, chat_history

# Initialize Model Parameters
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
hf_auth = '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'
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
embedding_model_kwargs = {"device": "cuda"}

llm = create_llm(model_id, hf_auth, device)
chain = create_retriever(llm, embedding_model, embedding_model_kwargs)

# Tkinter GUI
class ChatbotGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("AI Chatbot Interface")
        self.geometry("800x600")

        self.query_label = tk.Label(self, text="Enter your query:")
        self.query_label.pack(pady=10)

        self.query_entry = tk.Entry(self, width=100)
        self.query_entry.pack(pady=10)

        self.submit_button = tk.Button(self, text="Submit", command=self.submit_query)
        self.submit_button.pack(pady=10)

        self.output_label = tk.Label(self, text="Answer:")
        self.output_label.pack(pady=10)

        self.output_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=90, height=10)
        self.output_text.pack(pady=10)

        self.source_label = tk.Label(self, text="Source Documents:")
        self.source_label.pack(pady=10)

        self.source_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=90, height=10)
        self.source_text.pack(pady=10)

        self.chat_history = []

    def submit_query(self):
        query = self.query_entry.get()
        if query:
            answer, sources, self.chat_history = generate_answer(query, chain, self.chat_history)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, answer)

            self.source_text.delete(1.0, tk.END)
            for doc in sources:
                self.source_text.insert(tk.END, f"{doc}\n\n")

if __name__ == "__main__":
    app = ChatbotGUI()
    app.mainloop()
