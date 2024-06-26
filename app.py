import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
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

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)


        st.title("Sentiment Analysis")
        positive, negative, neutral = helper.sentiment_analysis(selected_user, df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Positive")
            st.title(positive)
        with col2:
            st.header("Negative")
            st.title(negative)
        with col3:
            st.header("Neutral")
            st.title(neutral)

        # Pie chart for sentiment
        fig, ax = plt.subplots()
        ax.pie([positive, negative, neutral], labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)

        st.title("Sentiment Over Time")
        df['date'] = pd.to_datetime(df['date'])
        sentiment_over_time = df.groupby('date')['sentiment'].apply(
            lambda x: (x == 'Positive').sum() - (x == 'Negative').sum()
        ).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sentiment_over_time['date'], sentiment_over_time['sentiment'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score (Positive - Negative)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Sentiment by user (if overall is selected)
        if selected_user == 'Overall':
            st.title("Sentiment by User")
            user_sentiment = df.groupby('user')['sentiment'].apply(
                lambda x: (x == 'Positive').sum() - (x == 'Negative').sum()
            ).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            user_sentiment.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        st.title("Sentiment Confidence")

        def get_sentiment_with_confidence(text):
            sentiment, confidence = sentiment_analyzer.analyze(text)
            return pd.Series([sentiment, confidence])

        df[['sentiment', 'confidence']] = df['message'].apply(get_sentiment_with_confidence)

        fig, ax = plt.subplots()
        df.groupby('sentiment')['confidence'].mean().plot(kind='bar', ax=ax)
        ax.set_ylabel('Average Confidence')
        ax.set_title('Average Sentiment Confidence')
        st.pyplot(fig)