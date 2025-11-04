from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

agent = create_agent(
    model=model,
)