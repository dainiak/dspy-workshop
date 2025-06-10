# Run with:
# python -m streamlit run streamlit_demo.py
import streamlit as st

from optimize_email_parser import load_training_data, EmailParser

st.title("📧 DSPy Email Parser Demo")
st.markdown("### See the power of DSPy optimization in action!")

# Load data
@st.cache_resource
def get_optimized_parser():
    parser = EmailParser()
    parser.load("optimized_parser.json")
    return parser

# Create parsers
if 'parser' not in st.session_state:
    with st.spinner("Optimizing parser with DSPy..."):
        st.session_state.parser = get_optimized_parser()

# Email input
email_text = st.text_area("Paste an email here:", height=200)

if st.button("Extract Action Items"):
    if email_text:
        pred = st.session_state.parser(email_text)
        dspy_result = {
            'intent': pred.intent,
            'action_items': [item.strip() for item in pred.action_items.split(';')],
            'deadlines': [d.strip() for d in pred.deadlines.split(';')],
            'priority': pred.priority
        }
        st.json(dspy_result)

with st.expander("Example Emails"):
    examples = load_training_data()
    for i, ex in enumerate(examples[:3]):
        st.text(f"Email {i+1}:")
        st.text(ex.email[:200] + "...")
        st.json({
            'intent': ex.intent,
            'action_items': ex.action_items.split(';'),
            'deadlines': ex.deadlines.split(';'),
            'priority': ex.priority
        })