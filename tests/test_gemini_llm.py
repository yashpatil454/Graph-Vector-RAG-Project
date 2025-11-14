# from google.generativeai as genai
import sys
from pathlib import Path
# Ensure project root is on sys.path BEFORE importing app.* modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.config import settings

# client = genai.Client()

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     google_api_key=settings.GOOGLE_API_KEY,
#     contents="How does AI work?"
# )
# print(response.text)

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=settings.GOOGLE_API_KEY)
response = llm.invoke("Sing a ballad of LangChain.")
response = llm.generate
print(response)