import pandas as pd
import streamlit as st
import pickle

# Load the trained pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Team and city options
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Title
st.title('🏏 IPL Win Predictor')

# Team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# City and Target
selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target Score')

# Match situation inputs
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_out = st.number_input('Wickets Out', min_value=0, max_value=10, step=1)

# Predict Button
if st.button('Predict Probability'):

    # Calculate features
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets = 10 - wickets_out
    crr = score / overs if overs != 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left != 0 else 0

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Display input
    st.subheader("Match Situation")
    st.table(input_df)

    # Predict win probability
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.subheader("Win Prediction 🔮")
    st.success(f"{batting_team} - {round(win * 100)}% chance to win")
    st.error(f"{bowling_team} - {round(loss * 100)}% chance to win")



