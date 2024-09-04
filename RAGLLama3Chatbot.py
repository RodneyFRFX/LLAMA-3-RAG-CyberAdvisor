# Imports
from torch import cuda, bfloat16
import transformers
import os
import re

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain


""" Methods """

def trim_answer(full_answer):
  """
  Cuts the answer down to just what is said after the model states "Helpful Answer:"
  
  Parameters
  ----------
  
  full_answer : str
    The full answer provided by the chatbot
    
  Returns
  -------
  
  str
    The trimmed down answer of everything after "Helpful Answer"  
  
  """
  # Find index of the "helpful answer"
  helpful_answer_index = full_answer.find("Helpful Answer")
 
  # Make sure "Helpful Answer" is in the answer"
  if helpful_answer_index != -1:
    # grab everything after "Helpful Answer"
    helpful_answer = full_answer[helpful_answer_index + len("Helpful Answer: "):]
    return helpful_answer
  else:
    print("Cannot find a suitable answer.")   
    
def create_llm(model_id, hf_auth, device):
  # Quantization configuration using bitsandbytes library
  # setting it to load a large model w/ less GPU memory
  bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
  )
  
  # Initialize Hugging face items w/ access token
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
  
  # Enable evaluation mode - allows model inference
  model.eval()

  # Prints if it makes it this far
  print(f"Model loaded on {device}")
  
  # Initialize the tokenizer
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth,
    padding_side="right"
  )

  # Add pad token as eos token
  tokenizer.pad_token_id = tokenizer.eos_token_id
  
  # Add in the pad_token_ids... fixes eos_token_id error
  model.generation_config.pad_token_id = tokenizer.pad_token_id
  
  # Initialize the terminators
  terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  # Initialize the generator parameter
  generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    eos_token_id=terminators,
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # max number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
  )
  
  # Implement Hugging Face Pipeline in LangChain
  llm = HuggingFacePipeline(pipeline=generate_text)
  
  return llm
  
def create_retriever(llm, model_name, model_kwargs):
  # Inititialize the embeddings model
  embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
  
  # Load in the faiss database
  vectorstore = FAISS.load_local("NewPhase2FaissIndex", embeddings, allow_dangerous_deserialization=True)
  
  # Initialize the conversational retrieval chain: aka creating memory
  chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

  return chain

def generate_answer(query, chain, chat_history):

  # Ask the chat bot a question and it will give an answer in the result chain
  result = chain({
    "question": query,
    "chat_history": chat_history,
  })
  
  # Find the important answer
  helpful_answer = trim_answer(result['answer'])
  
  # Grab the source documents
  source_docs = result['source_documents']
  
  # Place initial question and answer into the chat_history array
  chat_history = [(query, helpful_answer)] # Not sure if I should return the full answer or just the Helpful answer
  
  return helpful_answer, source_docs, chat_history


def main():
  """Initialize Model Parameters """

  # Initialize model and device
  model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
  device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

  # hf authentication token
  hf_auth = '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'

  # Initialize embedding model parameters
  embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
  embedding_model_kwargs = {"device": "cuda"}

  # Initialize chatting parameters
  chat_history = []

  # Initialize Letter's Regex
  ans_reg = re.compile('\A[A-DX]')
  
  # Initialize model and retriever
  llm = create_llm(model_id, hf_auth, device)
  chain = create_retriever(llm, embedding_model, embedding_model_kwargs)
  
  # Answer the question
  query = "How can we ensure that our physical security measures are integrated with our cybersecurity efforts, and that we're not creating any vulnerabilities through physical access points?"
  
  answer, sources, chat_history = generate_answer(query, chain, chat_history) 
  
  print(f"\nQuestion: {query}")
  print("Answer:", answer, "\n")
  
  print("Source Documents:")
  print(sources)
  
  # Ask the model a follow-up question if there is one
  follow_up_query = "Anything else to add?"
  
  follow_up_answer, new_sources, chat_history = generate_answer(follow_up_query, chain, chat_history)
  
  print(f"""\nFollow-up Question: {follow_up_query}""")
  print("Follow-up Answer:", follow_up_answer)

  print("Follow up Source Documents:")
  print(new_sources)
  

if __name__ == "__main__":
    main()


# Find which letter is involved
#letter_arr = ans_reg.findall(helpful_answer)
#letter = letter_arr[0] # First match

#print("Letter:", letter, "\n")



