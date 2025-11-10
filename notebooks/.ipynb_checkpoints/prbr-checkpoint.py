import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mplsoccer import Pitch
from highlight_text import fig_text
import os
import io
from matplotlib import font_manager

# Set page configuration
st.set_page_config(
    page_title="Football Post-Recovery Analysis",
    page_icon="⚽",
    layout="wide"
)

# App title and description
st.title("⚽ Premier League Post-Recovery Actions Analysis")
st.markdown("Analyze how players perform immediately after ball recoveries in the Premier League 2024-25")


@st.cache_data(show_spinner=True)
def load_data():
    """Load and preprocess data files"""
    # Load events data
    events_file = "PL_24_25_INT_BREAK.parquet"
    minutes_file = "PL_24_25_Mins_INT_BREAK.csv"
    
    events_data = pd.read_parquet(events_file)
    minutes_data = pd.read_csv(minutes_file)
    events_data['x'] = events_data['x']*1.2
    events_data['y'] = events_data['y']*.8
    events_data['endX'] = events_data['endX']*1.2
    events_data['endY'] = events_data['endY']*.8
    
    # Ensure minutes data has '90s_played' column
    if 'Mins' in minutes_data.columns:
        minutes_data['90s_played'] = minutes_data['Mins'] / 90
    
    return events_data, minutes_data

def analyze_post_recovery_actions(data, min_90s_played=0):
    """
    Analyze post-recovery actions including ball retention and progressive passes.
    
    Parameters:
    data (DataFrame): Event data with x, y coordinates and event types
    min_90s_played (float): Minimum number of 90-minute matches played to include player
    
    Returns:
    DataFrame: Player statistics for post-recovery actions
    """
    # Create shifted columns to link each event with the next event
    data['x_n'] = data['x'].shift(-1)
    data['y_n'] = data['y'].shift(-1)
    data['endX_n'] = data['endX'].shift(-1)
    data['endY_n'] = data['endY'].shift(-1)
    data['type_n'] = data['type'].shift(-1)
    data['outcomeType_n'] = data['outcomeType'].shift(-1)
    
    # Identify ball recoveries followed by various types of actions
    data['PR_pass'] = (data['type'] == 'BallRecovery') & \
                      ((data['type_n'] == 'Pass') | 
                       (data['type_n'] == 'TakeOn') | 
                       (data['type_n'] == 'Dispossessed') | 
                       (data['type_n'] == 'OffsidePass') | 
                       (data['type_n'] == 'Foul')) & \
                      ((data['outcomeType_n'] == 'Successful') | 
                       (data['outcomeType_n'] == 'Unsuccessful'))
    
    # Extract the post-recovery events
    PR_df = data[data['PR_pass'] == True].copy()
    
    # Calculate post-recovery progressive passes
    PR_df = calculate_progressive_passes(PR_df)
    
    # Create indicator columns for different types of post-recovery events
    PR_df['Successful_Pass_PR'] = (PR_df['type_n'] == 'Pass') & (PR_df['outcomeType_n'] == 'Successful')
    PR_df['Unsuccessful_Pass_PR'] = (PR_df['type_n'] == 'Pass') & (PR_df['outcomeType_n'] == 'Unsuccessful')
    PR_df['Dispossessed_PR'] = (PR_df['type_n'] == 'Dispossessed') & (PR_df['outcomeType_n'] == 'Successful')
    PR_df['Successful_TakeOn_PR'] = (PR_df['type_n'] == 'TakeOn') & (PR_df['outcomeType_n'] == 'Successful')
    PR_df['Unsuccessful_TakeOn_PR'] = (PR_df['type_n'] == 'TakeOn') & (PR_df['outcomeType_n'] == 'Unsuccessful')
    PR_df['OffsidePass_PR'] = (PR_df['type_n'] == 'OffsidePass')
    PR_df['Foul_PR'] = (PR_df['type_n'] == 'Foul')
    
    # Create a new column for successful progressive passes only
    PR_df['Successful_Progressive_Pass_PR'] = PR_df['progressive'] & PR_df['Successful_Pass_PR']
    
    # Group by player and calculate totals
    PR_df_grouped = PR_df.groupby(['player', 'team']).agg(
        Successful_Pass_PR=('Successful_Pass_PR', 'sum'),
        Unsuccessful_Pass_PR=('Unsuccessful_Pass_PR', 'sum'),
        Successful_TakeOn_PR=('Successful_TakeOn_PR', 'sum'), 
        Unsuccessful_TakeOn_PR=('Unsuccessful_TakeOn_PR', 'sum'),
        Dispossessed_PR=('Dispossessed_PR', 'sum'),
        OffsidePass_PR=('OffsidePass_PR', 'sum'),
        Foul_PR=('Foul_PR', 'sum'),
        Progressive_Pass_PR=('Successful_Progressive_Pass_PR', 'sum')  # Changed to only count successful progressive passes
    ).reset_index()
    
    # Get total ball recoveries per player
    total_recoveries = data[data['type'] == 'BallRecovery'].groupby(['player']).size().reset_index(name='Total_Recoveries')
    
    # Merge with the grouped dataframe
    PR_df_grouped = PR_df_grouped.merge(total_recoveries, on='player', how='left')
    
    # Calculate ball retention percentage
    PR_df_grouped['Ball_Retention_%'] = (
        PR_df_grouped['Successful_Pass_PR'] + PR_df_grouped['Successful_TakeOn_PR'] + PR_df_grouped['Foul_PR']
    ) / (
        PR_df_grouped['Successful_Pass_PR'] + PR_df_grouped['Unsuccessful_Pass_PR'] +
        PR_df_grouped['Dispossessed_PR'] + PR_df_grouped['OffsidePass_PR'] +
        PR_df_grouped['Unsuccessful_TakeOn_PR'] + PR_df_grouped['Successful_TakeOn_PR'] + PR_df_grouped['Foul_PR'] + 0.0001  # Add small value to prevent division by zero
    ) * 100
    
    # Calculate progressive passes percentage
    PR_df_grouped['%_Prog_Passes'] = (PR_df_grouped['Progressive_Pass_PR'] / (PR_df_grouped['Total_Recoveries'] + 0.0001)) * 100
    
    # Calculate the overall counts of actions by player
    overall_counts = data.groupby(['player', 'type']).size().unstack().reset_index().fillna(0)
    
    # Calculate the counts of successful and unsuccessful actions
    successful_counts = data[data['outcomeType'] == 'Successful'].groupby(['player', 'type']).size().unstack().add_prefix('Successful_').reset_index().fillna(0)
    unsuccessful_counts = data[data['outcomeType'] == 'Unsuccessful'].groupby(['player', 'type']).size().unstack().add_prefix('Unsuccessful_').reset_index().fillna(0)
    
    # Merge all counts into a single DataFrame
    player_stats = overall_counts.merge(successful_counts, on='player', how='left').merge(unsuccessful_counts, on='player', how='left').fillna(0)
    
    # Merge with the post-recovery stats
    merged_df = player_stats.merge(PR_df_grouped, on='player', how='inner')
    
    # Get the columns to display
    columns_to_display = [
        'player', 'team', 'Ball_Retention_%', 'Total_Recoveries',
        'Progressive_Pass_PR', '%_Prog_Passes', 'Successful_Pass_PR', 'Unsuccessful_Pass_PR',
        'Successful_TakeOn_PR', 'Unsuccessful_TakeOn_PR', 'Dispossessed_PR', 
        'OffsidePass_PR', 'Foul_PR'
    ]
    
    if 'BallRecovery' in merged_df.columns:
        columns_to_display.append('BallRecovery')
    
    final_df = merged_df[columns_to_display]
    
    return final_df, PR_df

def calculate_progressive_passes(df):
    """
    Calculate progressive passes - passes that move the ball significantly closer to the goal.
    Progressive passes are defined as passes that move the ball at least 25% closer to the goal.
    
    Parameters:
    df (DataFrame): Event data with x, y coordinates
    
    Returns:
    DataFrame: Original dataframe with progressive passes indicator
    """
    # Calculate the distance from the goal for the pass origin
    df['beginning'] = np.sqrt(np.square(120 - df['x_n']) + np.square(40 - df['y_n']))
    
    # Calculate the distance from the goal for the pass destination
    df['end'] = np.sqrt(np.square(120 - df['endX_n']) + np.square(40 - df['endY_n']))
    
    # A pass is progressive if it moves the ball at least 25% closer to the goal
    df['progressive'] = df['end'] / df['beginning'] < 0.75
    
    # Convert boolean to int for aggregation
    df['progressive'] = df['progressive'].astype(int)
    
    return df

def plot_post_recovery_passes(player_data, player_name, team_name, games_played, player_stats):
    """
    Plot post-recovery passes on a football pitch.
    
    Parameters:
    player_data (DataFrame): Event data filtered for a specific player
    player_name (str): Name of the player
    team_name (str): Name of the player's team
    games_played (float): Number of 90-minute games played
    player_stats (DataFrame): Player statistics including ball retention and progressive passes percentages
    
    Returns:
    matplotlib.figure.Figure: The resulting figure
    """
    # Get player stats from the stats dataframe
    player_row = player_stats[player_stats['player'] == player_name].iloc[0]
    ball_retention_pct = player_row['Ball_Retention_%']
    prog_passes_pct = player_row['%_Prog_Passes']
    
    # Filter for ball recoveries
    recov_data = player_data[player_data['type'] == 'BallRecovery']
    
    # Filter for successful passes after recovery
    succ_pass_afrecov = recov_data[
        (recov_data['type_n'] == 'Pass') & 
        (recov_data['outcomeType_n'] == 'Successful')
    ]
    
    # Reset index for the successful passes
    succ_pass_afrecov_reset = succ_pass_afrecov.reset_index()
    
    # Calculate progressive passes
    succ_pass_afrecov_reset = calculate_progressive_passes(succ_pass_afrecov_reset)
    succ_pass_afrecov_prog = succ_pass_afrecov_reset[succ_pass_afrecov_reset['progressive'] == 1]
    
    # Filter for unsuccessful passes after recovery
    unsucc_pass_afrecov = recov_data[
        (recov_data['type_n'] == 'Pass') & 
        (recov_data['outcomeType_n'] == 'Unsuccessful')
    ]
    
    # Reset index for the unsuccessful passes
    unsucc_pass_afrecov_reset = unsucc_pass_afrecov.reset_index()
    
    # Calculate progressive unsuccessful passes
    unsucc_pass_afrecov_reset = calculate_progressive_passes(unsucc_pass_afrecov_reset)
    unsucc_pass_afrecov_prog = unsucc_pass_afrecov_reset[unsucc_pass_afrecov_reset['progressive'] == 1]
    
    # Count the passes
    succ_PR = len(succ_pass_afrecov)
    unsucc_PR = len(unsucc_pass_afrecov)
    prog_PR = len(succ_pass_afrecov_prog)  # Only successful progressive passes
    
    # Count fouls
    foul_PR = len(recov_data[recov_data['type_n'] == 'Foul'])
    
    # Calculate per-90 stats
    succ_PR_per90 = succ_PR / games_played if games_played > 0 else 0
    unsucc_PR_per90 = unsucc_PR / games_played if games_played > 0 else 0
    prog_PR_per90 = prog_PR / games_played if games_played > 0 else 0
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15.5, 10))
    fig.set_facecolor('#242526')
    ax.patch.set_facecolor('#242526')
    
    # Draw the pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1e1e1e', line_color='#FFFFFF')
    pitch.draw(ax=ax)
    
    # Plot successful passes if they exist
    if not succ_pass_afrecov.empty:
        lc1 = pitch.lines(
            succ_pass_afrecov.x_n, succ_pass_afrecov.y_n,
            succ_pass_afrecov.endX_n, succ_pass_afrecov.endY_n,
            lw=4, transparent=True, comet=True, label='completed passes',
            color='#24a8ff', ax=ax
        )
        pitch.scatter(
            succ_pass_afrecov.endX_n, succ_pass_afrecov.endY_n, 
            s=70, marker='o', edgecolors='none', c='#24a8ff', 
            zorder=3, label='goal', ax=ax, alpha=1
        )
    
    # Plot progressive passes if they exist
    if not succ_pass_afrecov_prog.empty:
        lc2 = pitch.lines(
            succ_pass_afrecov_prog.x_n, succ_pass_afrecov_prog.y_n,
            succ_pass_afrecov_prog.endX_n, succ_pass_afrecov_prog.endY_n,
            lw=4, transparent=True, comet=True, label='progressive passes',
            color='#03fc24', ax=ax
        )
        pitch.scatter(
            succ_pass_afrecov_prog.endX_n, succ_pass_afrecov_prog.endY_n, 
            s=70, marker='o', edgecolors='none', c='#03fc24', 
            zorder=4, label='goal', ax=ax, alpha=1
        )
    
    # Plot unsuccessful passes if they exist
    if not unsucc_pass_afrecov.empty:
        lc3 = pitch.lines(
            unsucc_pass_afrecov.x_n, unsucc_pass_afrecov.y_n,
            unsucc_pass_afrecov.endX_n, unsucc_pass_afrecov.endY_n,
            lw=4, transparent=True, comet=True, label='unsuccessful passes',
            color='#FF5959', ax=ax, alpha=0.5
        )
        pitch.scatter(
            unsucc_pass_afrecov.endX_n, unsucc_pass_afrecov.endY_n, 
            s=70, marker='o', edgecolors='none', c='#FF5959', 
            zorder=1, label='goal', ax=ax, alpha=0.8
        )
    
    # Invert the y-axis
    plt.gca().invert_yaxis()
    
    # Add text annotations
    fig_text(
        0.516, 0.995, f"<{player_name}>", font='Arial Rounded MT Bold', size=30,
        ha="center", color="#FFFFFF", fontweight='bold', highlight_textprops=[{"color": '#FFFFFF'}]
    )
    fig_text(
        0.518, 0.941,
        f"Successful Passes after Ball Recovery | {team_name} | {games_played:.2f} 90s Played",
        font='Arial Rounded MT Bold', size=22,
        ha="center", color="#FFFFFF", fontweight='bold'
    )
    fig_text(
        0.518, 0.892,
        f"{ball_retention_pct:.1f}% of the Balls that {player_name.split()[0] if player_name else 'Player'} Recovers end up in a successful ACTION",
        font='Arial Rounded MT Bold', size=18,
        ha="center", color="#FFFFFF", fontweight='bold'
    )
    
    # Add progressive passes percentage
    #fig_text(
     #   0.518, 0.843,
      #  f"{prog_passes_pct:.1f}% of Recoveries by {player_name.split()[0] if player_name else 'Player'} Result in a Progressive Pass",
       # font='Arial Rounded MT Bold', size=18,
        #ha="center", color="#FFFFFF", fontweight='bold'
    #)
    
    fig_text(
        0.770, 0.105, "Made by: @pranav_m28\nData: Opta\n2024-25", 
        font='Arial Rounded MT Bold', size=18,
        ha="center", color="#FFFFFF", fontweight='bold'
    )
    fig_text(
        0.265, 0.105, 
        f"<Successful Passes = {succ_PR} ({succ_PR_per90:.2f})>\n<Progressive Passes = {prog_PR} ({prog_PR_per90:.2f})>\n<Unsuccessful Passes = {unsucc_PR} ({unsucc_PR_per90:.2f})>",
        font='Arial Rounded MT Bold', size=16,
        ha="center", color="#FFFFFF", fontweight='bold', 
        highlight_textprops=[{"color": '#24a8ff'}, {"color": '#03fc24'}, {"color": "#FF5959"}]
    )
    
    # Set plot style
    plt.style.use("dark_background")
    mpl.rc('axes', edgecolor='#131313', linewidth=1.2)
    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#1e1e1e' 
    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'
    
    # Add direction of play indicator
    fig_text(
        0.510, 0.10, "Direction of Play", font='Arial Rounded MT Bold', size=18,
        ha="center", color="#FFFFFF", fontweight='bold'
    )
    plt.arrow(49.2, -3, 20, 0, fc='#FFFFFF', ls='-', lw=1.9, head_length=1, head_width=1)
    
    return fig

def analyze_post_recovery_actions(data, min_90s_played=0):
    """
    Analyze post-recovery actions including ball retention and progressive passes.
    
    Parameters:
    data (DataFrame): Event data with x, y coordinates and event types
    min_90s_played (float): Minimum number of 90-minute matches played to include player
    
    Returns:
    DataFrame: Player statistics for post-recovery actions
    """
    # Create shifted columns to link each event with the next event
    data['x_n'] = data['x'].shift(-1)
    data['y_n'] = data['y'].shift(-1)
    data['endX_n'] = data['endX'].shift(-1)
    data['endY_n'] = data['endY'].shift(-1)
    data['type_n'] = data['type'].shift(-1)
    data['outcomeType_n'] = data['outcomeType'].shift(-1)
    
    # Identify ball recoveries followed by various types of actions
    data['PR_pass'] = (data['type'] == 'BallRecovery') & \
                      ((data['type_n'] == 'Pass') | 
                       (data['type_n'] == 'TakeOn') | 
                       (data['type_n'] == 'Dispossessed') | 
                       (data['type_n'] == 'OffsidePass') | 
                       (data['type_n'] == 'Foul')) & \
                      ((data['outcomeType_n'] == 'Successful') | 
                       (data['outcomeType_n'] == 'Unsuccessful'))
    
    # Extract the post-recovery events
    PR_df = data[data['PR_pass'] == True].copy()
    
    # Calculate post-recovery progressive passes
    PR_df = calculate_progressive_passes(PR_df)
    
    # Create indicator columns for different types of post-recovery events
    PR_df['Successful_Pass_PR'] = (PR_df['type_n'] == 'Pass') & (PR_df['outcomeType_n'] == 'Successful')
    PR_df['Unsuccessful_Pass_PR'] = (PR_df['type_n'] == 'Pass') & (PR_df['outcomeType_n'] == 'Unsuccessful')
    PR_df['Dispossessed_PR'] = (PR_df['type_n'] == 'Dispossessed') & (PR_df['outcomeType_n'] == 'Successful')
    PR_df['Successful_TakeOn_PR'] = (PR_df['type_n'] == 'TakeOn') & (PR_df['outcomeType_n'] == 'Successful')
    PR_df['Unsuccessful_TakeOn_PR'] = (PR_df['type_n'] == 'TakeOn') & (PR_df['outcomeType_n'] == 'Unsuccessful')
    PR_df['OffsidePass_PR'] = (PR_df['type_n'] == 'OffsidePass')
    PR_df['Foul_PR'] = (PR_df['type_n'] == 'Foul')
    
    # Create a new column for successful progressive passes only
    PR_df['Successful_Progressive_Pass_PR'] = PR_df['progressive'] & PR_df['Successful_Pass_PR']
    
    # Group by player and calculate totals
    PR_df_grouped = PR_df.groupby(['player', 'team']).agg(
        Successful_Pass_PR=('Successful_Pass_PR', 'sum'),
        Unsuccessful_Pass_PR=('Unsuccessful_Pass_PR', 'sum'),
        Successful_TakeOn_PR=('Successful_TakeOn_PR', 'sum'), 
        Unsuccessful_TakeOn_PR=('Unsuccessful_TakeOn_PR', 'sum'),
        Dispossessed_PR=('Dispossessed_PR', 'sum'),
        OffsidePass_PR=('OffsidePass_PR', 'sum'),
        Foul_PR=('Foul_PR', 'sum'),
        Progressive_Pass_PR=('Successful_Progressive_Pass_PR', 'sum')  # Changed to only count successful progressive passes
    ).reset_index()
    
    # Get total ball recoveries per player
    total_recoveries = data[data['type'] == 'BallRecovery'].groupby(['player']).size().reset_index(name='Total_Recoveries')
    
    # Merge with the grouped dataframe
    PR_df_grouped = PR_df_grouped.merge(total_recoveries, on='player', how='left')
    
    # Calculate ball retention percentage
    PR_df_grouped['Ball_Retention_%'] = (
        PR_df_grouped['Successful_Pass_PR'] + PR_df_grouped['Successful_TakeOn_PR'] + PR_df_grouped['Foul_PR']
    ) / (
        PR_df_grouped['Successful_Pass_PR'] + PR_df_grouped['Unsuccessful_Pass_PR'] +
        PR_df_grouped['Dispossessed_PR'] + PR_df_grouped['OffsidePass_PR'] +
        PR_df_grouped['Unsuccessful_TakeOn_PR'] + PR_df_grouped['Successful_TakeOn_PR'] + PR_df_grouped['Foul_PR'] + 0.0001  # Add small value to prevent division by zero
    ) * 100
    
    # Calculate progressive passes percentage
    PR_df_grouped['%_Prog_Passes'] = (PR_df_grouped['Progressive_Pass_PR'] / (PR_df_grouped['Total_Recoveries'] + 0.0001)) * 100
    
    # Calculate the overall counts of actions by player
    overall_counts = data.groupby(['player', 'type']).size().unstack().reset_index().fillna(0)
    
    # Calculate the counts of successful and unsuccessful actions
    successful_counts = data[data['outcomeType'] == 'Successful'].groupby(['player', 'type']).size().unstack().add_prefix('Successful_').reset_index().fillna(0)
    unsuccessful_counts = data[data['outcomeType'] == 'Unsuccessful'].groupby(['player', 'type']).size().unstack().add_prefix('Unsuccessful_').reset_index().fillna(0)
    
    # Merge all counts into a single DataFrame
    player_stats = overall_counts.merge(successful_counts, on='player', how='left').merge(unsuccessful_counts, on='player', how='left').fillna(0)
    
    # Merge with the post-recovery stats
    merged_df = player_stats.merge(PR_df_grouped, on='player', how='inner')
    
    # Get the columns to display
    columns_to_display = [
        'player', 'team', 'Ball_Retention_%', 'Total_Recoveries',
        'Progressive_Pass_PR', '%_Prog_Passes', 'Successful_Pass_PR', 'Unsuccessful_Pass_PR',
        'Successful_TakeOn_PR', 'Unsuccessful_TakeOn_PR', 'Dispossessed_PR', 
        'OffsidePass_PR', 'Foul_PR'
    ]
    
    if 'BallRecovery' in merged_df.columns:
        columns_to_display.append('BallRecovery')
    
    final_df = merged_df[columns_to_display]
    
    return final_df, PR_df

# Main app logic
try:
    # Load data from system files
    events_data, minutes_data = load_data()
    
    # Filter options in sidebar
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Minutes played filter
        min_90s_played = st.slider(
        "Minimum 90s played:", 
        min_value=0.0, 
        max_value=float(round(minutes_data['90s_played'].max())) if '90s_played' in minutes_data.columns else 10.0,
        value=1.0,
        step=0.5
)
        
        # Team filter
        available_teams = sorted(events_data['team'].unique())
        selected_teams = st.multiselect(
            "Select teams:",
            available_teams,
            default=available_teams[:5] if len(available_teams) > 5 else available_teams
        )
        
        # Apply team filter to events data
        if selected_teams:
            filtered_events = events_data[events_data['team'].isin(selected_teams)]
        else:
            filtered_events = events_data
    
    # Analysis section
    st.header("📈 Player Analysis")
    
    # Analyze the event data
    player_stats, PR_data = analyze_post_recovery_actions(filtered_events)
    
    # Merge player stats with minutes data based on player name and team
    if '90s_played' in minutes_data.columns:
        # Ensure team column is in the same format in both dataframes
        merged_stats = player_stats.merge(
            minutes_data[['player', 'team', '90s_played']], 
            on=['player', 'team'], 
            how='left'
        ).fillna(0)
        
        # Filter by minimum 90s played
        filtered_stats = merged_stats[merged_stats['90s_played'] >= min_90s_played]
        
        # Calculate per 90 columns
        filtered_stats['PR_per_90'] = filtered_stats['Total_Recoveries'] / filtered_stats['90s_played']
        filtered_stats['Prog_PR_per_90'] = filtered_stats['Progressive_Pass_PR'] / filtered_stats['90s_played']
        
        # Sort by ball retention %
        sorted_stats = filtered_stats.sort_values('Ball_Retention_%', ascending=False)
        
        # Display the table with the most relevant columns first
        display_columns = [
            'player', 'team', '90s_played', 'Ball_Retention_%', '%_Prog_Passes', 
            'Total_Recoveries', 'PR_per_90', 'Progressive_Pass_PR', 'Prog_PR_per_90',
            'Successful_Pass_PR', 'Unsuccessful_Pass_PR', 'Successful_TakeOn_PR', 
            'Unsuccessful_TakeOn_PR', 'Dispossessed_PR', 'OffsidePass_PR', 'Foul_PR'
        ]
        
        display_columns = [col for col in display_columns if col in sorted_stats.columns]
        
        # Format the table
        formatted_stats = sorted_stats[display_columns].copy()
        formatted_stats['Ball_Retention_%'] = formatted_stats['Ball_Retention_%'].round(1)
        formatted_stats['%_Prog_Passes'] = formatted_stats['%_Prog_Passes'].round(1)
        formatted_stats['90s_played'] = formatted_stats['90s_played'].round(2)
        formatted_stats['PR_per_90'] = formatted_stats['PR_per_90'].round(2)
        formatted_stats['Prog_PR_per_90'] = formatted_stats['Prog_PR_per_90'].round(2)
        
        # Set index to 1-based row numbers instead of 0-based
        formatted_stats.index = range(1, len(formatted_stats) + 1)
        
        # Display the table
        st.dataframe(
            formatted_stats,
            use_container_width=True,
            height=400,
            column_config={
                'Ball_Retention_%': st.column_config.NumberColumn(
                    "Ball Retention %",
                    format="%.1f%%",
                    help="Percentage of post-recovery actions that are successful"
                ),
                '%_Prog_Passes': st.column_config.NumberColumn(
                    "Prog. Pass %",
                    format="%.1f%%",
                    help="Progressive passes as percentage of total recoveries"
                ),
                'PR_per_90': st.column_config.NumberColumn(
                    "Recoveries/90",
                    help="Ball recoveries per 90 minutes"
                ),
                'Prog_PR_per_90': st.column_config.NumberColumn(
                    "Prog. Passes/90",
                    help="Progressive passes after recovery per 90 minutes"
                ),
            }
        )
        
        # Player visualization section
        st.header("Player Pass Map")
        
        # Get available players based on current filters
        available_players = sorted(filtered_stats['player'].unique())
        
        if available_players:
            # Player selection
            selected_player = st.selectbox(
                "Select a player to view post-recovery passing map:",
                available_players
            )
            
            # Get player details
            player_row = filtered_stats[filtered_stats['player'] == selected_player].iloc[0]
            player_team = player_row['team']
            player_90s = player_row['90s_played']
            
            # Filter data for the selected player
            player_data = filtered_events[filtered_events['player'] == selected_player]
            
            # Create and display visualization
            if not player_data.empty:
                fig = plot_post_recovery_passes(
                    player_data,
                    selected_player,
                    player_team,
                    player_90s,
                    filtered_stats  # Pass the player stats to use the correct Ball_Retention_% value
                )
                st.pyplot(fig)
                
                # Add download button
                buf = io.BytesIO()  # Use io.BytesIO() instead of plt.io.BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)

                st.download_button(
                    label="Download Pass Map",
                    data=buf,
                    file_name=f"{selected_player}_post_recovery_passes.png",
                    mime="image/png"
                )
            else:
                st.warning("No data available for the selected player.")
        else:
            st.warning("No players match the current filters. Try adjusting the minimum 90s played or team selection.")
    else:
        st.error("Minutes data format issue. Please ensure it contains 'player' and 'Mins' columns.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    #st.info("Make sure the data files 'PL_24_25_INT_BREAK.csv' and 'PL_24_25_Mins_INT_BREAK.csv' are in the same directory as this script.")


    with st.sidebar:
        st.markdown("---")
        st.markdown("### Connect with me")
        st.markdown("- 🐦 [Twitter](https://twitter.com/pranav_m28)")
        st.markdown("- 🔗 [GitHub](https://github.com/pranavm28)")
        st.markdown("- ❤️ [BuyMeACoffee](https://buymeacoffee.com/pranav_m28)")
    
# Footer
st.markdown("---")
st.markdown("Created by @pranav_m28 | Data source: Opta | Premier League 2024-25 | Updated till GW29")
