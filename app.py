import streamlit as st
from models.rag_pipeline import rag_answer

st.set_page_config(page_title="Jupiter FAQ Bot", page_icon="💬")
st.title("💬 Jupiter FAQ Bot")
st.write("Ask any question about Jupiter's banking services. The bot will find the best FAQ and answer conversationally!")

user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):
        result = rag_answer(user_query)
    st.markdown("---")
    st.subheader("Best Match from FAQ")
    st.markdown(f"**Q:** {result['retrieved_faq']['question']}")
    st.markdown(f"**A:** {result['retrieved_faq']['answer']}")
    st.subheader("Bot's Answer")
    st.markdown(result['llm_response'])
    st.markdown("---")
    st.info("Powered by FAISS, Sentence Transformers, and HuggingFace LLMs.") 