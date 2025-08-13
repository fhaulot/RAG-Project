from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded properly
if not API_KEY:
    raise Exception('GOOGLE_API_KEY not found. Please set it in your .env file.')

# Use the API key
client = genai.configure(api_key=API_KEY)
google_chosen_model='gemini-2.5-flash-lite'
model = genai.GenerativeModel(google_chosen_model)

question = 'I am sick, I sent an email to my main coach and my campus coordinator, what else should I do?'
context = "If you cannot attend class for any reason, usually when you are sick. There are 2 steps: Warn your day coach (main coach or co-coach) and campus coordinator by email. Example emails: main.coach@becode.org co.coach@becode.org campus.coordinator@becode.org. You have to have a justification paper, upload it to moodle. If it is too late to upload it, send it to your campus coordinator and your main coach."
prompt = f'Use the following snippet:\n {context}\n\n To answer this question: {question}'
print("Prompt:\n",prompt)

response = model.generate_content(contents=prompt)
print("\n\nAnswer:\n", response.text)


# calling prompt embedding
prompt_embed = genai.embed_content(
    model="models/embedding-001",
    content='I am sick, I sent an email to my main coach and my campus coordinator, what else should I do?'
)

print(prompt_embed['embedding'])