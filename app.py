import os
from pathlib import Path
import dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from st_audiorec import st_audiorec
import openai



TMP = 0
MODEL = 'gpt-3.5-turbo-instruct'

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ['API_KEY']
# Creating the Streamlit App
st.title('Reimaging your interaction with money')
st.header("Welcome to OMNI. Open Money's Neural Interace.")
st.text('Write out your goal or speak away your tasks to OMNI to see the magic of OMNI unfold!')

system_prompt_template = """You are a assistant to the Chief Financial Officer well versed with converting vague or specific task from them into an ordered list of tasks to be done.
You have to respond to the CFO's work request with a list of tasks that have to be done in a very specific order by workers of the CFO's company to accomplish the request efficiently.
Take your time to think very clearly and generate a list of tasks which have to be done sequentially to achieve the output.
Think it step by step. For eg. if you have to pay some money to a person, you need to have one of the task as: Find the account details of the <Person's name given by the CFO> for the payout from the database of the company.
You have to be very specific and always make the list with less than 4 tasks. Be comprehensive with each item of the list to make sure 4 people can simulataneously pick up the tasks and execute them without blocking one another.
The final output should read like a paragraph even though it strictly is formatted as a list.

DO NOT output anything except a list of at maximum 4 mutually exclusive tasks in a numbered list starting with the first task to be done
DO NOT at all cost output more than 4 tasks.

If you can, keep the list of tasks less than 4

This is the request by the CFO : {request}
"""
# title_template = PromptTemplate(
#     input_variables  = ['topic']
#     template = 'write a note on {topic}'
# )

base_prompt = PromptTemplate.from_template(system_prompt_template)
llm = OpenAI(model=MODEL, temperature=TMP)
inp = st.text_input("Enter the task you want OMNI to do for you")
# Record audio and transcribe it
wav_audio_data = st_audiorec()
transcript = ''
if wav_audio_data is not None:
    # st.audio(wav_audio_data, format='audio/wav')
    with open("audio.wav", "wb") as f:
        f.write(wav_audio_data)
        f.close()

    file = open('audio.wav', "rb")
    transcript = openai.audio.transcriptions.create(
    model = 'whisper-1',
    file=file,
    response_format='text',
    temperature=0.8
    )
st.divider()
st.subheader('Command to OMNI')
st.text(st.write(transcript))
# print(wav_audio_data)
# print(file)

# # request = st.text_input('Enter a prompt to statisy your need')
if not inp:
    request = transcript
    llm_chain_base = LLMChain(prompt=base_prompt, llm=llm)
    response_base = llm_chain_base.run(request)

    st.write(response_base)

else:
    request = inp
    llm_chain_base = LLMChain(prompt=base_prompt, llm=llm)
    response_base = llm_chain_base.run(request)

    st.write(response_base)
# st.subheader('Chief Tasker')
# llm_chain_final = LLMChain(prompt=checker_prompt, llm=llm)
# response_final = llm_chain_final.run(response_base)
# st.write(response_final)