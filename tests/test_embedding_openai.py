# from openai import OpenAI

import sys
from pathlib import Path

# Ensure project root is on sys.path BEFORE importing app.* modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.config import settings
# client = OpenAI(api_key=settings.OPENAI_API_KEY)

# client.embeddings.create(
#   model="text-embedding-3-small",
#   input="The food was delicious and the waiter...",
#   encoding_format="float"
# )

# response = client.responses.create(
#     model="gpt-5",
#     reasoning={"effort": "low"},
#     instructions="Talk like a pirate.",
#     input="Are semicolons optional in JavaScript?",
# )

# print(response.output_text)