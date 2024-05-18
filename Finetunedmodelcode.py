from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
import csv
import torch
from transformers import BitsAndBytesConfig

# Step 1: Load and process documents from a CSV file
csv_files = ['Farmers_Protests.csv']
loaders = [CSVLoader(file_path=file_path, encoding="utf-8") for file_path in csv_files]
documents = []
for loader in loaders:
    data = loader.load()  # Retrieve data from the loader
    documents.extend(data)

# Split documents into chunks of 1000 characters with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Step 2: Create a vector store using FAISS
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings(model_name=modelPath)
db = FAISS.from_documents(docs, embeddings)

# Step 3: Create a retriever to fetch context documents
retriever = db.as_retriever(search_kwargs={"k": 25})

# Step 4: Configure the language model and create a pipeline
tokenizer = AutoTokenizer.from_pretrained("path")

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("path",
                                             device_map='cuda',
                                             quantization_config=nf4_config)

question_answerer = pipeline(
    return_full_text=True,
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
)

llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.9, "max_length": 512},
)

# Step 5: Define a valid custom prompt template
template = """
Please generate an answer to the following question using any relevant information from the provided context:

Context: {context}

Question: {question}

Helpful Answer:
"""

custom_rag_prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

# Helper function to convert documents into a usable string format
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 6: Create a sequence of operations for the chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Step 7: Invoke the chain with a question
question = "Generate the first 100 words of an eye-catching article on farmer's protests"

try:
    responses = []
    for i in range(500):
        print(f"printing this iteration--------------{i}----------------")
        result = qa_chain.invoke(question)
        print(f"Answer: {result}")
        responses.append({"ID": i + 1, "Response": result})
    
    with open("responses1.csv", "w", newline="") as csvfile:
        fieldnames = ["ID", "Response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for response in responses:
            writer.writerow(response)

except AssertionError as e:
    print(f"Error: {str(e)}")
