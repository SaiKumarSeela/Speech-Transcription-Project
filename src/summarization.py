from src.utils import get_transcript_using_assemblyai
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import assemblyai as aai
from src.constants import SUMMARIZATION_MODEL
load_dotenv()

# assembly_api_key = os.getenv("ASSEMBLYAI_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")


# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="Llama3-8b-8192"
# )

def summarise_transcript(groq_api_key,transcript):

    llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name= SUMMARIZATION_MODEL
    )
    # Prepare the prompt for summarization
    summarise_prompt = f""" Summarise the following transcript delimited by 3 backticks without any introductory phrases: {transcript} """

    # Create the chat message structure for Groq API
    messages = [
        {
            'role': 'system',
            'content': 'You are a helpful assistant who summarises the provided text concisely in no more than 1000 words.'
        },
        {
            'role': 'user',
            'content': summarise_prompt,
        },
    ]

    # Get the response from the Llama model
    response = llm.invoke(messages)

    # print(type(response))

    # Extract and print only the content from the response
    summary_content = response.content
    summary_content = summary_content.split(":", 1)[-1].strip() 
    # print("Summary:", summary_content)
    return summary_content


# Example usage
# if __name__ == "__main__":
#     mp3file_path = "Samples/chunk1.mp3" 
#     summarise_transcript(llm, mp3file_path)