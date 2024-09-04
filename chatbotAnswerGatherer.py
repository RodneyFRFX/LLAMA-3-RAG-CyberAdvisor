# -*- coding: utf-8 -*-
# Imports
from torch import cuda, bfloat16
import transformers
import pandas as pd
import os
import re

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# Initialize model and device
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Quantization configuration using bitsandbytes library
# setting it to load a large model w/ less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# Initialize Hugging face items w/ access token
hf_auth = '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'
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

"""## Methods"""

# Finds the "Helpful Answer" out of the full answer the model provides
def trim_answer(full_answer):
  # Find index of the "helpful answer"
  helpful_answer_index = full_answer.find("Helpful Answer")
 
  # Make sure "Helpful Answer" is in the answer"
  if helpful_answer_index != -1:
    # grab everything after "Helpful Answer"
    helpful_answer = full_answer[helpful_answer_index + len("Helpful Answer: "):]
    return helpful_answer
  else:
    print("Cannot find a suitable answer.")    

"""## Initializing the tokenizer"""

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth,
    padding_side="right"
)

# Add pad token as eos token
tokenizer.pad_token_id = tokenizer.eos_token_id

"""## Deciding the Stopping Criteria"""

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Add in the pad_token_ids... fixes eos_token_id error
model.generation_config.pad_token_id = tokenizer.pad_token_id

"""## Initialize Parameters"""

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    eos_token_id=terminators,
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=50,  # max number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
)

"""## Implement Hugging Face Pipeline in LangChain """
llm = HuggingFacePipeline(pipeline=generate_text)

""" Grab the Vector embeddings """
model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectorstore = FAISS.load_local("NewPhase2FaissIndex", embeddings, allow_dangerous_deserialization=True)

"""## Creating Memory"""

# Initialize the conversational retrieval chain
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

"""## Asking Questions"""

# Load questions into a dictionary of dataframes for each sheet
df = pd.read_excel('Q&A.xlsx', sheet_name='task-mcq-mitre', usecols='C')

# Concatinate both dataframes together
#df = pd.concat(sheets_dict.values(), ignore_index=True);

# Initialize Letter's Regex
ans_reg = re.compile('\A[A-DX]')

# Loop through and ask the chatbot all the questions, then add them to a new dataframe
for index, row in df.iterrows():
  # Initialize chatting parameters
  chat_history = []

  # Ask the chatbot a question and it will give an answer in the result chain
  query = row['Prompt']
  result = chain({
      "question": query,
      "chat_history": chat_history
  })
  
  # Find index of the "helpful answer"
  helpful_answer_index = result['answer'].find("Helpful Answer")

  # Make sure "Helpful Answer" is in the answer"
  if helpful_answer_index != -1:
    # grab everything after "Helpful Answer"
    helpful_answer = result['answer'][helpful_answer_index + len("Helpful Answer: "):]
  else:
    print("Cannot find a suitable answer.")
    
  # Find answer
  answer_arr = re.findall(ans_reg, helpful_answer)
  # Make sure answer is there, if not grab full answer
  if answer_arr:
    answer = answer_arr[0] # Grab first answer
  else:
    answer = helpful_answer
  print(answer)
  
  # Place chatbot answers into the dataframe
  df.at[index, 'RAG meta-llama/Meta-Llama-3-8B-Instruct Answers'] = answer
  
# Save DF back to an Excel File
df.to_excel('Updated Phase 1 Responses/Phase2 check_mitre Q&A.xlsx', index=False)
