from langchain.text_splitter import TextSplitter
from langchain.document_loaders import PyPDFLoader
import tiktoken



tokenizer = tiktoken.get_encoding('p50k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

pdf = PyPDFLoader('test.pdf').load()
     

text_splitter = TextSplitter(
    chunk_size=512,
    length_function=tiktoken_len,
    seperators=['\n\n', '\n', ' ']
    )

text_splitter.split(pdf)