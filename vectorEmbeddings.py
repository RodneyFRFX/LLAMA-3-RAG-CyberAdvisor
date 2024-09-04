from torch import cuda, bfloat16
import transformers
import os
import bs4

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
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# Enable evaluation mode - allows model inference
model.eval()

# Prints if it makes it this far
print(f"Model loaded on {device}")

"""## Methods"""

def json_metadata_func(record: dict, metadata: dict, cveID) -> dict:
  metadata["cveID"] = cveID
  return metadata


# Opens all of the files and folders of json files, grabs and returns the loaded files    
def open_everything(folder_path, exclude_files):
  # Iterate over all items in given folder
  for item in os.listdir(folder_path):
    # Get full path of item opened
    item_path = os.path.join(folder_path, item)
    
    # Check if item is excluded
    if item in exclude_files:
      continue
      
    # If another folder, recursively call function on that directory
    if os.path.isdir(item_path):
      open_everything(item_path, exclude_files)
    # If it is a file, then decide what file type it is and read it
    else: 
      try:
        if bool(json_reg.search(item_path)):
          # Grab the id by cutting off .json from the item
          cveID = item[:-5]
        
          # Load the json file with a json loader, grabbing the value in the descriptions key
          json_loader = JSONLoader(
            file_path = item_path,
            jq_schema = '.containers.cna.descriptions[].value',
            text_content=False,
            metadata_func = lambda record, metadata: json_metadata_func(record, metadata, cveID)
          )
               
          json_doc = json_loader.load()
        
          # Place json docs into the array
          all_documents.extend(json_doc)
          
        elif bool(pdf_reg.search(item_path)):
          pdf_loader = PyPDFLoader(item_path)
          pages = pdf_loader.load_and_split()
            
          # Place pdf pages into the array
          all_documents.extend(pages)
          
        elif bool(html_reg.search(item_path)):
          html_loader = BSHTMLLoader(item_path)
          html_doc = html_loader.load()
          
          # Place html docs into array (includes htm)
          all_documents.extend(html_doc)
          
        elif bool(docx_reg.search(item_path)):
          docx_loader = Docx2txtLoader(item_path)
          docx_doc = docx_loader.load()
          
          # Place docx docs into array
          all_documents.extend(docx_doc)
          
        elif bool(xlsx_reg.search(item_path)):
          xlsx_loader = UnstructuredExcelLoader(item_path, mode="elements")
          xlsx_doc = xlsx_loader.load()
          
          # Place excel docs into array
          all_documents.extend(xlsx_doc)
          
        elif bool(csv_reg.search(item_path)):
          csv_loader = CSVLoader(file_path=item_path)
          csv_data = csv_loader.load()
          
          # Place csv data into array
          all_documents.extend(csv_data)
          
        else:
          loader = TextLoader(item_path)
          doc = loader.load()
          
          # Place general docs (usually txt) into the array
          all_documents.extend(doc)
          
      except Exception as e:
        print(f"Error opening the file {item_path}: {e}")
        continue
        
  return
  
# Accepts a file path, splits the cves into urls or pdfs
def open_csv_links(file_path, link_col):
  # Read all files and place them in the appropriate array
  with open(file_path, newline ='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      if bool(pdf_reg.search(row[link_col])):
        pdfs.append(row[link_col])
      else:
        urls.append(row[link_col])
  return


"""Document Loading"""

from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, BSHTMLLoader, Docx2txtLoader, UnstructuredExcelLoader, CSVLoader, TextLoader
import csv
import re
import nest_asyncio

# Make sure nest_asyncio is applied
nest_asyncio.apply()

# Initizlize Sequence of Urls and pdfs
urls = []
pdfs = []

# Initialize document array
all_documents = []

#Initialize the Regex
pdf_reg = re.compile('.+\.pdf$')
html_reg = re.compile('.+\.html?$') #l is optional
json_reg = re.compile('.+\.json$')
docx_reg = re.compile('.+\.docx$')
xlsx_reg = re.compile('.+\.xlsx$')
csv_reg = re.compile('.+\.csv$')

# Read all CSV links
open_csv_links('Data Sources/Cert Text Links.csv', 'url')
open_csv_links('Data Sources/Cisa Links.csv', 'url')
open_csv_links('Top 20 Common Crawl.csv', 'url')

# Feed urls into WebBaseLoader to gain page_content and the meta_data
#loader = WebBaseLoader(
#  urls,
#  bs_kwargs=dict(
#    parse_only=bs4.SoupStrainer(
#      class_=("post_content","post-title")
#    )
#  ),
#  default_parser='lxml'
#)

loader = WebBaseLoader(urls, continue_on_failure=True, default_parser='lxml')
web_documents = loader.aload()

# Feed pdfs into PyPDFLoader and then split into pages.
pdf_pages = []

for pdf in pdfs:
  try:
    pdf_loader = PyPDFLoader(pdf)
    pages = pdf_loader.load_and_split()
    pdf_pages.extend(pages)
  except Exception as e:
    print(f"Error opening the PDF {pdf}: {e}")
    continue

# Grab the cve documents
#open_everything("/home/ztl1776/LLama2Chatbot/cvelistV5/cves/", ['delta.json', 'deltaLog.json'])

# Grab the cyber docs provided by a professional
#open_everything("/home/ztl1776/LLama2Chatbot/cyber docs/", ['urls']) 

"""## Document Splitting"""

# Combine the rest of the Documents
for doc in web_documents:
  all_documents.append(doc)

for pdf_page in pdf_pages:
  all_documents.append(pdf_page)
   
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# Create the splits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
all_splits = text_splitter.split_documents(all_documents)

print(all_splits[0:100])
print("Number of splits:", len(all_splits)) 

"""## Create and Store Vector Embeddings"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
extension = FAISS.from_documents(all_splits, embeddings)

# Load old database
vectorstore = FAISS.load_local("New_Phase1_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Combine the index extension to the vectorstore
vectorstore.merge_from(extension) # Not in FAISS GPU

# Save the FAISS index locally
vectorstore.save_local("New_Phase2_faiss_index")
