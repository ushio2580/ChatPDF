import streamlit as st
import os
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embedding.openai import OpenAIEmbeddings
from langchain.vextorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langChain.callbacks import get_openai_callback

'''import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader'''

#Sidebar content
with st.sidebar:
    st.title("Streamlit Langchain App")
    st.markdown('''**This is a simple Streamlit app that uses Langchain to answer questions based on a given text.**')
    st.markdown('---')
    st.subheader('About')'' 
                
                
                
                
                
                
                
                ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Anmol Patel](https://twitter.com/AnmolPatel1)')




    def main():
        st.title("Chat with your PDF")   


        load_dotenv()


        #upload a pdf file
        pdf=st.file_uploader("Upload your PDF",type="pdf")
        st.write(pdf.name)

        #st.write(pdf)
        if pdf is None:
            pdf_reader = PdfReader(pdf)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text()
            text_splitter=RecursiveCharacterTextSplitter(
                chunck_size=1000,
                chunck_overlap=200,
                length_function=len
            )    
            chunks=text_splitter.split_text(text_text)


        
            store_name=pdf.name[:-4]

            if  os.path.exist(f"{store_name}.pkl","wb"):
                with open(f"{store_name}.pkl","wb","rb") as f:
                    VectorStore=pickle.load(f)
               # st.write("Embedding loaded from the disk")    
            else: 
                    #embeddings
                embeddings=OpenAIEmbeddings()

            #vector store
                VectorStore=FAISS.from_texts(chunks,embeddings=embeddings)       
                with open(f"{store_name}.pkl","wb") as f:
                    pickle.dump(VectorStore,f)
               #  st.write("Embeding computation complete")      

            #Accept user question
            query=st.text_input("ask question about your PDF file:")
            st.write(query)


            if query:
                docs=VectorStore.similarity_search(query=query,k=3)
                llm=OpenAI(model_name='gpt-3.5-turbo')

                chain-load_qa_chain(llm=llm,chain_type="stuff")
                with get_openai_callback() as cb:
                     response=chain.run(input_documents=docs,question=query)
                     print(cb)

                st.write(response)
                



           # st.write(text)
       

if __name__ == '__main__':  
    main()
