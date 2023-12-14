
import streamlit as st
import pandas as pd
import re
import os
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emoji import demojize
import librosa
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import speech_recognition as sr
import numpy as np
from pydub import AudioSegment
from urlextract import URLExtract
import seaborn as sns
extract = URLExtract()

# Function to process chat data
def process_chat_data(uploaded_file):
    # Read the uploaded file
     # Read the uploaded file
    chat_data = uploaded_file.read().decode('utf-8')

    # Determine the date format (12-hour or 24-hour)
    if re.search(r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s[APM]+', chat_data):
        date_format = "12-hour"
        date_format_str = '%m/%d/%y, %I:%M %p'
    else:
        date_format = "24-hour"
        date_format_str = '%m/%d/%y, %H:%M'

    # Define the regular expression pattern based on the detected format
    if date_format == "24-hour":
        pattern = r'(\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2})\s-\s(.+?):\s(.+)'
    else:
        pattern = r'(\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s[APM]+)\s-\s(.+?):\s(.+)'

    # Find all matches using the regular expression
    matches = re.findall(pattern, chat_data)

    # Extract data from matches
    dates = [match[0] for match in matches]  # Keep the full date-time string
    senders = [match[1] for match in matches]
    messages = [match[2] for match in matches]

    # Create a DataFrame with the extracted data
    data = {
        'Date-Time': dates,
        'Sender': senders,
        'Message': messages
    }

    df = pd.DataFrame(data)

    # Use the determined format to convert 'Date-Time' column
    df['Date-Time'] = pd.to_datetime(df['Date-Time'], format=date_format_str)

    # Extract 'No of Hours' and 'No of Minutes'
    df['No of Hours'] = df['Date-Time'].dt.hour
    df['No of Minutes'] = df['Date-Time'].dt.minute

    # Calculate additional columns
    df['Recipient'] = df['Sender'].shift(-1)
    df['Word Count'] = df['Message'].apply(lambda x: len(str(x).split()))
    return df

    

# Function to extract emojis from the message
def extract_emojis(message):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    emojis = emoji_pattern.findall(message)
    return emojis

# Function to calculate the sentiment score using VADER
def get_sentiment_score_vader(message):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(message)['compound']

# Function to calculate the sentiment dictionary using VADER
def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict

# Function for overall analysis
def overall_analysis(df):
    # Create the SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Calculate the sentimental score of each message using VADER
    df['Sentiment Score'] = df['Message'].apply(get_sentiment_score_vader)

    # Calculate 'Overall sentiment dictionary is', 'Negative', 'Neutral', 'Positive', and 'Sentence Overall Rated As'
    df['Overall sentiment dictionary is'] = df['Message'].apply(sentiment_scores)
    df['Negative'] = df['Overall sentiment dictionary is'].apply(lambda x: x['neg'])
    df['Neutral'] = df['Overall sentiment dictionary is'].apply(lambda x: x['neu'])
    df['Positive'] = df['Overall sentiment dictionary is'].apply(lambda x: x['pos'])
    df['Overall Sentiment'] = df['Sentiment Score'].apply(lambda score: 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral')

    # Calculate the percentage of positive, negative, and neutral sentimental scores
    df['Positive percentage'] = df['Positive'] * 100
    df['Negative percentage'] = df['Negative'] * 100
    df['Neutral percentage'] = df['Neutral'] * 100

    # Convert percentage columns into integer percentages
    df['Positive percentage'] = df['Positive percentage'].astype(int)
    df['Negative percentage'] = df['Negative percentage'].astype(int)
    df['Neutral percentage'] = df['Neutral percentage'].astype(int)

    # Extract emojis from the messages
    df['Emojis'] = df['Message'].apply(extract_emojis)

    return df

# Function for analysis of selected sender
def sender_analysis(df, selected_sender):
    filtered_df = df[df['Sender'] == selected_sender]

    # Create the SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Calculate the sentimental score of each message using VADER
    filtered_df['Sentiment Score'] = filtered_df['Message'].apply(get_sentiment_score_vader)

    # Calculate 'Overall sentiment dictionary is', 'Negative', 'Neutral', 'Positive', and 'Sentence Overall Rated As'
    filtered_df['Overall sentiment dictionary is'] = filtered_df['Message'].apply(sentiment_scores)
    filtered_df['Negative'] = filtered_df['Overall sentiment dictionary is'].apply(lambda x: x['neg'])
    filtered_df['Neutral'] = filtered_df['Overall sentiment dictionary is'].apply(lambda x: x['neu'])
    filtered_df['Positive'] = filtered_df['Overall sentiment dictionary is'].apply(lambda x: x['pos'])
    filtered_df['Overall Sentiment'] = filtered_df['Sentiment Score'].apply(lambda score: 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral')

    # Calculate the percentage of positive, negative, and neutral sentimental scores
    filtered_df['Positive percentage'] = filtered_df['Positive'] * 100
    filtered_df['Negative percentage'] = filtered_df['Negative'] * 100
    filtered_df['Neutral percentage'] = filtered_df['Neutral'] * 100

    # Convert percentage columns into integer percentages
    filtered_df['Positive percentage'] = filtered_df['Positive percentage'].astype(int)
    filtered_df['Negative percentage'] = filtered_df['Negative percentage'].astype(int)
    filtered_df['Neutral percentage'] = filtered_df['Neutral percentage'].astype(int)

    # Extract emojis from the messages
    filtered_df['Emojis'] = filtered_df['Message'].apply(extract_emojis)

    return filtered_df

# Function to generate graphs
def fetch(analysis_type,df):
    if analysis_type != 'Overall Analysis':
        df = df[df['Sender'] ==analysis_type]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['Message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['Message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['Message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)
def generate_graphs(df):
    activity_graph(df)
    line_graph(df)
    percentage_contribution(df)
    create_heatmap(df,'Positive')
    create_heatmap(df,'Negative')
    create_heatmap(df,'Neutral')
    generate_word_cloud(df, 'Positive')
    generate_word_cloud(df, 'Negative')
    generate_word_cloud(df, 'Neutral')
    most_common_words(df, 'Positive')
    most_common_words(df, 'Neutral')
    most_common_words(df, 'Negative')
    most_active_user(df, 'Positive')
    most_active_user(df, 'Neutral')
    most_active_user(df, 'Negative')
def activity_graph(df):
    sentiment_counts = df.groupby(['Overall Sentiment', 'Date-Time']).size().reset_index(name='Message Count')

    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'blue'
    }

    for sentiment in ['Positive', 'Negative', 'Neutral']:
        sentiment_data = sentiment_counts[sentiment_counts['Overall Sentiment'] == sentiment]

        if not sentiment_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sentiment_data['Date-Time'], y=sentiment_data['Message Count'], name=f'{sentiment} Sentiment'))

            # Set the bar color
            fig.update_traces(marker_color=color_map[sentiment])

            # Set the title and axis labels
            fig.update_layout(title=f'{sentiment} Sentiment Activity Over Time')
            fig.update_xaxes(title_text='Date-Time')
            fig.update_yaxes(title_text='Message Count')

            # Hide the text inside the bars
            fig.update_traces(textposition='outside', insidetextfont_color='rgba(0,0,0,0)')

            # Increase the width and height
            fig.update_layout(
                width=700,  # Set the width
                height=600   # Set the height
            )

            st.plotly_chart(fig)
        else:
            st.markdown("")
def line_graph(df):
    sentiment_counts = df.groupby(['Overall Sentiment', 'Date-Time']).size().reset_index(name='Message Count')

    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'blue'
    }

    fig = go.Figure()

    # Check if there is data for any sentiment
    data_available = False
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        sentiment_data = sentiment_counts[sentiment_counts['Overall Sentiment'] == sentiment]
        if not sentiment_data.empty:
            data_available = True
            fig.add_trace(go.Scatter(x=sentiment_data['Date-Time'], y=sentiment_data['Message Count'], mode='lines', name=f'{sentiment} Sentiment', line=dict(color=color_map[sentiment])))

    if not data_available:
        st.markdown("No data available for any sentiment.")
    else:
        # Set the title
        fig.update_layout(title='Sentiment Activity Over Time')

        # Set the axis labels
        fig.update_xaxes(title_text='Date-Time')
        fig.update_yaxes(title_text='Message Count')

        # Set the size of the graph
        fig.update_layout(
            width=820,  # Set the width
            height=650   # Set the height
        )

        st.plotly_chart(fig)




import plotly.express as px

# Function to create Percentage Contribution (Positive, Neutral, Negative)
def percentage_contribution(df):
    total_count = len(df)
    sentiment_counts = df['Overall Sentiment'].value_counts()
    percentages = (sentiment_counts / total_count) * 100

    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'blue'
    }

    fig = px.pie(values=percentages.values, names=percentages.index, title='Percentage Contribution (Positive, Neutral, Negative')
    fig.update_traces(marker=dict(colors=[color_map[sentiment] for sentiment in percentages.index]))

    st.plotly_chart(fig)

def create_heatmap(df, title):

    # Filter the dataframe by sentiment
    sentiment_df = df[df['Overall Sentiment'] == title]

    # Check if there is data available
    if not sentiment_df.empty:
        st.markdown(f"<span style='font-size: 20px;'>**{title} Sentiment Heatmap**</span>", unsafe_allow_html=True)
        # Create a figure with subplots
        fig, ax = plt.subplots(figsize=(8, 4))

        # Create a heatmap for the specified sentiment
        heatmap = sns.heatmap(sentiment_df.corr(), annot=True, cmap="YlGnBu", ax=ax)

        # Decrease the font size of x-axis and y-axis tick labels
        heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=6)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=6)

        # Create a border around the subplot
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

        st.pyplot(fig)
    else:
        st.markdown("")
# Function to generate Word Cloud for a given sentiment category
def generate_word_cloud(df, sentiment):
    sentiment_df = df[df['Overall Sentiment'] == sentiment]
    words = ' '.join(sentiment_df['Message'])
    
    # Check if there are words to create a word cloud
    if words.strip() and sentiment != 'Negative':  # Ensure there are non-empty words and not 'Negative'
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(words)
        
        st.markdown(f"<span style='font-size: 20px;'>**Word Cloud for {sentiment} Sentiment.**</span>", unsafe_allow_html=True)

        # Create a bordered section with a smaller size and a red border
        with st.markdown("<div style='border: 100px solid red; padding: 1px; width: 400px;'>"):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
    
            # Display the Matplotlib figure using st.pyplot
            st.pyplot(fig)
    else:
        st.markdown("")

        
# # Function to generate Word Cloud for a given sentiment category
# def generate_word_cloud(df, sentiment):
#     sentiment_df = df[df['Overall Sentiment'] == sentiment]
#     words = ' '.join(sentiment_df['Message'])
    
#     # Check if there are words to create a word cloud
#     if words:
#         wordcloud = WordCloud(width=400, height=200, background_color='white').generate(words)
        
#         st.markdown(f"<span style='font-size: 20px;'>**Word Cloud for {sentiment} Sentiment.**</span>", unsafe_allow_html=True)

#         # Create a bordered section with a smaller size and a red border
#         with st.markdown("<div style='border: 100px solid red; padding: 1px; width: 400px;'>"):
#             fig, ax = plt.subplots(figsize=(8, 4))
#             ax.imshow(wordcloud, interpolation='bilinear')
#             ax.axis('off')
    
#             # Display the Matplotlib figure using st.pyplot
#             st.pyplot(fig)
#     else:
#         st.markdown("")




# Function to create a bar chart showing the most common words for a given sentiment category
def most_common_words(df, sentiment):
    sentiment_df = df[df['Overall Sentiment'] == sentiment]
    words = ' '.join(sentiment_df['Message']).split()
    word_counts = Counter(words)
    
    most_common = word_counts.most_common(10)
    
    if most_common:
        st.markdown(f"<span style='font-size: 20px;'>**Most Common Words for {sentiment} Sentiment'.**</span>", unsafe_allow_html=True)
        words, counts = zip(*most_common)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(words, counts)
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_xticklabels(words, rotation=45)
    
        st.pyplot(fig)
    else:
          st.markdown("")


# Function to find the most active user for a given sentiment category
def most_active_user(df, sentiment):
    sentiment_df = df[df['Overall Sentiment'] == sentiment]
    
    if not sentiment_df.empty:
        most_active_sender = sentiment_df['Sender'].value_counts().idxmax()
        sender_count = sentiment_df['Sender'].value_counts().max()
    
        st.write(f'Most Active User for {sentiment} Sentiment: {most_active_sender}')
        st.write(f'Number of Messages by {most_active_sender}: {sender_count}')
    else:
         st.markdown("")


# Define the desired columns for the output
# Define the desired columns for the output
desired_columns = ['Date-Time', 'No of Hours', 'No of Minutes', 'Sender', 'Recipient', 'Message', 'Word Count', 'Emojis', 'Positive percentage', 'Negative percentage', 'Neutral percentage', 'Overall Sentiment']

#AUDIO ANALYSIS

def calculate_intensity(audio_clip):
    # Export the AudioSegment to a temporary WAV file
    audio_clip.export("temp.wav", format="wav")

    # Load the temporary WAV file using librosa
    y, sr = librosa.load("temp.wav")

    # Calculate intensity as the root mean square (RMS) of the audio
    intensity = np.sqrt(np.mean(y**2))

    # Round off the intensity to 3 decimal places
    intensity = round(intensity, 3)

    # Remove the temporary WAV file
    os.remove("temp.wav")

    return intensity

def process_audio(audio_file):
    intensity_analyzer = SentimentIntensityAnalyzer()
    r = sr.Recognizer()

    audio = AudioSegment.from_file(audio_file)
    duration = len(audio) / 1000  # Convert to seconds

    clip_duration = 5  # Split audio into 5-second clips (you can adjust this)

    clip_data = []
    clip_times = []
    clip_products = []
    clip_sentiments = []

    for clip_number, start_time in enumerate(range(0, int(duration * 1000), clip_duration * 1000), 1):
        end_time = min(start_time + (clip_duration * 1000), int(duration * 1000))
        clip = audio[start_time:end_time]

        # Calculate intensity of audio clip using librosa
        clip_intensity = calculate_intensity(clip)

        # Extract text from audio clip
        clip_text = extract_text_from_audio(clip, r)

        # Calculate overall sentiment of audio clip
        clip_sentiment = calculate_sentiment(clip_text)

        # Calculate product of intensity and sentiment without standardization
        clip_product = round(clip_intensity * clip_sentiment, 3)

        clip_data.append([clip_number, clip_sentiment, clip_intensity, clip_product])
        clip_times.append(start_time / 1000)
        clip_products.append(clip_product)
        clip_sentiments.append(clip_sentiment)

    # Create a DataFrame for the table
    df = pd.DataFrame(clip_data, columns=["Clip Number", "Clip Sentiment", "Intensity", "Clip Product"])

    st.table(df)

    # Plot the intensity vs. clip number graph
    plot_intensity_vs_clip_number(range(1, len(clip_data) + 1), [d[2] for d in clip_data])

    # Plot the sentiment vs. clip number graph
    plot_sentiment_vs_clip_number(range(1, len(clip_data) + 1), [d[1] for d in clip_data])

    # Plot the intensity x sentiment vs. time graph
    plot_intensity_x_sentiment_vs_time(clip_times, clip_products)

    # Calculate the average of clip products
    average_product = round(np.mean(clip_products), 3)

    return average_product

def extract_text_from_audio(audio_clip, recognizer):
    with sr.AudioFile(audio_clip.export(format="wav")) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError:
        text = ""
    return text

def calculate_sentiment(text):
    intensity_analyzer = SentimentIntensityAnalyzer()
    sentiment = intensity_analyzer.polarity_scores(text)['compound']
    return sentiment

def plot_intensity_vs_clip_number(clip_numbers, intensity_values):
    plt.figure(figsize=(10, 6))
    plt.plot(clip_numbers, intensity_values, marker='o', color='green')
    plt.xlabel('Clip Number')
    plt.ylabel('Intensity')
    plt.title('Intensity vs. Clip Number')
    st.pyplot(plt)

def plot_sentiment_vs_clip_number(clip_numbers, sentiment_values):
    plt.figure(figsize=(10, 6))
    plt.plot(clip_numbers, sentiment_values, marker='o', color='orange')
    plt.xlabel('Clip Number')
    plt.ylabel('Sentiment')
    plt.title('Sentiment vs. Clip Number')
    st.pyplot(plt)

def plot_intensity_x_sentiment_vs_time(clip_times, intensity_x_sentiment_values):
    plt.figure(figsize=(10, 6))
    plt.plot(clip_times, intensity_x_sentiment_values, marker='o', color='blue')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Intensity x Sentiment')
    plt.title('Intensity x Sentiment vs. Time')
    st.pyplot(plt)

# The Streamlit app
def main():
    # Set a title for the web app
    st.sidebar.title("WhatsApp Chat Sentiment Analyzer")
    uploaded_file = st.sidebar.file_uploader('Choose the File', type=['txt', 'mp3', 'wav'], accept_multiple_files=True)

    if uploaded_file:
        for file in uploaded_file:
            if file.type == 'text/plain':
                st.markdown("<h1 style='text-align: center; color: white;'>WhatsApp Chat Sentiment Analyzer</h1>", unsafe_allow_html=True)
                df = process_chat_data(file)

                # Get unique senders (phone numbers and names)
                unique_senders = df['Sender'].unique().tolist()

                # Sorting
                unique_senders.sort()

                # Insert "Overall" at index 0
                unique_senders.insert(0, "Overall Analysis")

                analysis_type = st.sidebar.selectbox("Show analysis wrt", unique_senders)

                if st.sidebar.button("Chat Analysis"):
                    if analysis_type == "Overall Analysis":
                        df = overall_analysis(df)
                    else:  # Check if a specific sender is selected
                        df = sender_analysis(df, analysis_type)

                    num_messages, words, num_media_messages, num_links = fetch(analysis_type, df)
                    st.title("Top Statistics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.header("Total Messages")
                        st.title(num_messages)
                    with col2:
                        st.header("Total Words")
                        st.title(words)
                    with col3:
                        st.header("Media Shared")
                        st.title(num_media_messages)
                    with col4:
                        st.header("Links Shared")
                        st.title(num_links)

                    st.dataframe(df[desired_columns].head(10))
                    generate_graphs(df)  # Generate graphs for analysis

            elif file.type in ['audio/wav', 'audio/mp3']:
                st.markdown("<h1 style='text-align: center; color: white;'>Audio Sentiment Analyzer</h1>", unsafe_allow_html=True)
                st.audio(file, format="audio/wav/mp3")
                if st.sidebar.button("Audio Analysis"):
                        average_product = process_audio(file)
                        if average_product > 0:
                            st.markdown("<h2 style='color: white;'>Overall Sentiment:Positive</h2>", unsafe_allow_html=True)
                            
                        elif average_product < 0:
                            st.markdown("<h2 style='color: white;'>Overall Sentiment:Negative</h2>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h2 style='color: white;'>Overall Sentiment:Neutral</h2>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
