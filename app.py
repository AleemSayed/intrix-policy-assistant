import streamlit as st
from groq import RateLimitError
from rag import answer_question
import re


# Page configuration
st.set_page_config(
    page_title="Intrix Policy Assistant",
    page_icon="📚",
    layout="centered"
)


# Application header
st.title("📚 Intrix Policy Assistant")

st.caption(
    "Ask questions about company leave, expenses "
    "and IT security policies."
)

st.divider()


# Sidebar
with st.sidebar:
    st.header("About")

    st.write(
        "This AI assistant answers employee questions "
        "using company policy documents."
    )

    st.info(
        "Answers are generated only from the "
        "available policy documents."
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources below assistant answers
        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("📄 View sources"):
                    for source in sources:
                        st.write(
                            f"• {source['file']} "
                            f"(Page {source['page']})"
                        )


# Chat input
question = st.chat_input(
    "Ask a question about company policies..."
)


if question:

    # Display and save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching company policies..."):

            try:
                result = answer_question(question)

                answer = result["answer"]

                # Fix currency and HTML formatting
                answer = re.sub(r"[^\x00-\x7F]+(?=\s*\d[\d,]*)", " INR ", answer)
                answer = re.sub(r"<br\s*/?>", "\n", answer, flags=re.IGNORECASE)

                sources = result["sources"]

                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("📄 View sources"):
                        for source in sources:
                            st.write(
                                f"• {source['file']} "
                                f"(Page {source['page']})"
                            )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except RateLimitError:
                error_message = (
                    "Groq API rate limit reached. "
                    "Please wait a little and try again."
                )

                st.warning(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })

            except Exception as error:
                error_message = (
                    f"Something went wrong: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                })