Description
Create a FastAPI/Django app for this task.


Create 2 api endpoints
/upload 
Where upload above file 
Write a logic to extract text from the document, save in vector db and use it for RAG with chat agent
/chat
Ask question and get answer from the LLM 

OR you can create only 1 api endpoint which can take a file and ask questions if you can create this one only.
Link of the Document - THE AETHERIA PROTOCOL
You can use any framework for agentic workflow like LangChain or LangGraph.
For embeddings you can use SentenceTransformers or if you have any other third party api like gemini, openai etc.  you can use it.
You can use FAISS, ChromaDB for vectors. If you want to use vector db as Pinecone or Qdrant we will provide the api keys. 

We will provide the LLM key of Groq for LLM integration. If you have any other third party LLM provider like Gemini, OpenAi etc. you can use it.

Create a DockerFile for locally deploy it (Bonus point)

RAG pipeline
imports --
upload doc : save in folders all document
extract text
split into small data
embeddings
store in db
send context and questions to the LLM
result
