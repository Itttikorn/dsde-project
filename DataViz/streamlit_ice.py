import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sentence_transformers import SentenceTransformer
import numpy as np
import ast
from sklearn.preprocessing import normalize

torch.classes.__path__ = []

# Set page configuration
st.set_page_config(
    page_title="Scorpus Data",  # Title of the web page
    page_icon=":books:",         # Icon on the browser tab
    layout="wide",               # Set the layout to 'wide' for more space
    initial_sidebar_state="expanded"  # Optional: Set sidebar state (expanded/collapsed)
)

#------------------------------------- ----------------------- Model ------------------------------------------------------------
@st.cache_data
def load_Model():
    # Load the model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    df_pandas = pd.read_parquet("../ML/data_with_embeddings.parquet")
    df_pandas["embedding"] = df_pandas["embedding"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float16) if isinstance(x, str) else x)

    embeddings = np.array(df_pandas["embedding"].tolist()).astype('float32')
    
    embeddings_normalized = normalize(np.array(df_pandas["embedding"].tolist()).astype('float32'))
    return model, df_pandas, embeddings, embeddings_normalized

model, df_pandas, embeddings, norm_embeddings = load_Model()

#------------------------------------------------------------ Data ------------------------------------------------------------
@st.cache_data
def load_Data():
# Load the data file
    file_path = '../DataEn/data_cleaned.csv'
    data = pd.read_csv(file_path)

    # Preprocess data: count papers by subject
    data['subject_codes'] = data['subject_codes'].str.strip("[]").str.replace("'", "")
    data['subject_codes'] = data['subject_codes'].str.split(', ')
    papers_per_subject = data.explode('subject_codes')['subject_codes'].value_counts()
    return data, papers_per_subject

data, papers_per_subject = load_Data()

#------------------------------------------------------------ Streamlit App ------------------------------------------------------------
def main():
    st.markdown("<h1 style='text-align: center;'>Scorpus Data</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>in 2011-2023</h3>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📈 Data", "Search"])

    # Plot the number of papers per subject using Matplotlib
    with tab1:
        col1, col2 = st.columns([0.15, 0.75])
        with col2:
            st.subheader("Visualization: Papers Per Subject")

        col1, col2, col3 = st.columns([0.16, 0.6,0.2])
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 4))  # Adjust the size to be smaller
            papers_per_subject.plot(kind='bar', ax=ax2, color='lightblue', edgecolor='black', alpha=0.7)
            ax2.set_title("Number of Papers per Subject", fontsize=14)
            ax2.set_xlabel("Subject Code", fontsize=12)
            ax2.set_ylabel("Number of Papers", fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)

    with tab2:
        # Search section
        col1, col2 = st.columns([0.1, 0.9])
        with col2:
            st.subheader("Search Related Papers")

        col1, col2, col3, col4 = st.columns([0.15, 0.55,0.1, 0.1])
        with col2:
            text = st.text_area(
                "Enter sentence or word:",
                height=80
            )

        with col3:
            st.write("")  # Empty line for spacing
            st.write("")  # Empty line for spacing
            st.write("")  # Add more empty lines as needed for better alignment
            search_button = st.button("Search", use_container_width=True)
            st.write("")  # Empty line for spacing

        # Add slider for similarities and input box for number of papers
        col1, col2, col3, col4 = st.columns([0.15, 0.55,0.1, 0.1])
        with col2:
            similarity_range = st.slider(
                "Select Similarity Percentage:", 
                0, 100, (30, 100))
        with col3:
            num_papers = st.number_input(
                "number of papers:", 
                min_value=1, 
                max_value=100, 
                value=10
            )

        if search_button:
            if text.strip():
                query = text.strip()
                query_embedding = model.encode(query).astype('float32')
                query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize query embedding

                # Compute cosine similarity
                cosine_similarities = np.dot(norm_embeddings, query_embedding)

                # Filter and sort results
                similarity_threshold = similarity_range[0] / 100  # Convert to decimal
                results = [
                    (i, score) for i, score in enumerate(cosine_similarities) if score >= similarity_threshold
                ]
                results = sorted(results, key=lambda x: x[1], reverse=True)[:num_papers]

                # Prepare data for table
                if results:
                    table_data = []
                    years = []
                    subjects = []

                    for idx, score in results:
                        row = df_pandas.iloc[idx]
                        published_year = pd.to_datetime(row['published_date'], errors='coerce').year
                        table_data.append({
                            "Title": row['title'],
                            "Subject": row['subject_codes'],
                            "Year": published_year,
                            "Cited": int(row['citedby_count']),
                        })
                        years.append(published_year)
                        subjects.extend(row['subject_codes'])
                    # Convert to DataFrame and display as a table
                    results_df = pd.DataFrame(table_data)

                    col1, col2 = st.columns([0.1, 0.9])
                    with col2:
                        st.subheader("Search Results")

                    col1, col2, col3 = st.columns([0.18, 0.57, 0.25])
                    with col2:
                        st.dataframe(results_df, use_container_width=True) # Display the table

                    col1, col2,col3,col4, col5 = st.columns([0.07, 0.40, 0.06, 0.40, 0.07])
                    with col2:
                        # Plot the number of papers per year
                        year_counts = pd.Series(years).value_counts().sort_index()
                        st.subheader("Number of Papers in Each Year")
                        fig_year, ax_year = plt.subplots(figsize=(8, 4))  # Adjust the size to be smaller
                        year_counts.plot(kind='bar', ax=ax_year, color='lightgreen', edgecolor='black', alpha=0.7)
                        ax_year.set_title("Number of Papers per Year", fontsize=14)
                        ax_year.set_xlabel("Year", fontsize=12)
                        ax_year.set_ylabel("Number of Papers", fontsize=12)
                        ax_year.tick_params(axis='x', rotation=45)
                        st.pyplot(fig_year)

                    with col4:
                        # Plot the number of papers per subject
                        subject_counts = pd.Series(subjects).value_counts()
                        st.subheader("Number of Papers in Each Subject")
                        fig_subject, ax_subject = plt.subplots(figsize=(8, 4))  # Adjust the size to be smaller
                        subject_counts.plot(kind='bar', ax=ax_subject, color='lightcoral', edgecolor='black', alpha=0.7)
                        ax_subject.set_title("Number of Papers per Subject", fontsize=14)
                        ax_subject.set_xlabel("Subject", fontsize=12)
                        ax_subject.set_ylabel("Number of Papers", fontsize=12)
                        ax_subject.tick_params(axis='x', rotation=45)
                        st.pyplot(fig_subject)

                else:
                    st.write("No results found for the given input.")
            else:
                st.warning("Please enter a sentence or word to search!")


if __name__ == "__main__":
    main()