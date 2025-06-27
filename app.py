import streamlit as st
from models.rag_pipeline import rag_answer
from langdetect import detect
from translate import Translator

st.set_page_config(page_title="Jupiter Money FAQ Bot", page_icon="💬")
st.title("Jupiter Money FAQ Bot")
st.write("Ask any question about Jupiter's banking services. The bot will find the best FAQ and answer conversationally!")

if 'user_query' not in st.session_state:
    st.session_state['user_query'] = ''
if 'detected_lang' not in st.session_state:
    st.session_state['detected_lang'] = 'en'
if 'translated_answer' not in st.session_state:
    st.session_state['translated_answer'] = ''

user_query = st.text_input("Ask a question:", value=st.session_state['user_query'], key="user_query_input")

if user_query:
    # Detect language
    try:
        detected_lang = detect(user_query)
    except Exception:
        detected_lang = 'en'
    st.session_state['detected_lang'] = detected_lang
    # Translate to English if needed
    query_for_rag = user_query
    if detected_lang != 'en':
        translator = Translator(to_lang='en', from_lang=detected_lang)
        try:
            query_for_rag = translator.translate(user_query)
        except Exception:
            query_for_rag = user_query
    with st.spinner("Thinking..."):
        result = rag_answer(query_for_rag, return_prompt=True)
    st.subheader("Bot's Answer")
    st.markdown(result['llm_response'])
    # Option to translate answer back to original language
    if detected_lang != 'en':
        if st.button(f"Translate answer to {detected_lang}"):
            translator = Translator(to_lang=detected_lang, from_lang='en')
            try:
                translated = translator.translate(result['llm_response'])
            except Exception:
                translated = "[Translation failed] " + result['llm_response']
            st.session_state['translated_answer'] = translated
        if st.session_state['translated_answer']:
            st.markdown(f"**Translated Answer:** {st.session_state['translated_answer']}")
    # Related questions as clickable suggestions
    if result['related_questions']:
        st.markdown("**Related questions:**")
        cols = st.columns(len(result['related_questions']))
        for i, q in enumerate(result['related_questions']):
            if cols[i].button(q, key=f"related_{i}"):
                st.session_state['user_query'] = q
                st.session_state['translated_answer'] = ''
                st.experimental_rerun()
    with st.expander("Show thinking process (retrieved FAQ, prompt, etc.)"):
        st.markdown("---")
        st.subheader("Best Match from FAQ")
        st.markdown(f"**Q:** {result['retrieved_faq']['question']}")
        st.markdown(f"**A:** {result['retrieved_faq']['answer']}")
        st.subheader("System Prompt to LLM")
        st.code(result['system_prompt'], language='markdown')
        st.markdown("---")
    st.info("Powered by FAISS, Sentence Transformers, and HuggingFace LLMs.") 