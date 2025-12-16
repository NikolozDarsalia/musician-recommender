import streamlit as st
import requests
import time
from streamlit_sortables import sort_items
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
IMAGES_FOLDER = Path("artist_images")  # Folder for artist images

# Pastel color scheme
st.set_page_config(
    page_title="Music Recommendations",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for minimalistic pastel design with lavender background
st.markdown(
    """
    <style>
    .main {
        background-color: #e6e6fa;
        max-width: 900px;
        margin: 0 auto;
    }
    .stApp {
        background-color: #e6e6fa;
        max-width: 100vw;
        overflow-x: hidden;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #b3d9ff !important;
        border-radius: 12px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
    }
    .element-container {
        background-color: transparent !important;
    }
    .stButton>button {
        background-color: #b4a7d6;
        color: #000000;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #9d8ec4;
        box-shadow: 0 4px 12px rgba(180, 167, 214, 0.3);
    }
    .recommendation-card {
        background: #b3d9ff;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        border-left: 6px solid #b4a7d6;
        color: #000000;
        text-align: center;
    }
    .recommendation-mini {
        background: #b3d9ff;
        padding: 15px;
        border-radius: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        color: #000000;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .title {
        color: #000000;
        font-size: 2.5em;
        font-weight: 300;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #000000;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
        color: #000000 !important;
    }
    .rank-badge {
        background: #7d6fa8;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    .sortable-container {
        background: #e6e6fa;
        padding: 20px;
        border-radius: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "artists" not in st.session_state:
    st.session_state.artists = []
if "sorted_artists" not in st.session_state:
    st.session_state.sorted_artists = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "current_rec_index" not in st.session_state:
    st.session_state.current_rec_index = 0
if "shown_recommendations" not in st.session_state:
    st.session_state.shown_recommendations = []


def get_artist_image(artist_name):
    """Get artist image path if it exists"""
    # Try to find image with artist name
    image_path = IMAGES_FOLDER / f"{artist_name}.jpg"
    if image_path.exists():
        return str(image_path)
    # Try PNG
    image_path = IMAGES_FOLDER / f"{artist_name}.png"
    if image_path.exists():
        return str(image_path)
    return None


def fetch_artists():
    """Fetch available artists from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/survey-artists")
        if response.status_code == 200:
            data = response.json()
            artists_dict = data.get("artists", {})
            # Convert dict to list of artists
            artists_list = []
            for artist_id, artist_info in artists_dict.items():
                artists_list.append(
                    {
                        "artist_id": artist_info["artist_id"],
                        "artist_name": artist_info["name"],
                    }
                )
            return artists_list
        else:
            st.error(f"Failed to fetch artists: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return []


def submit_ratings(user_id, sorted_artists):
    """Submit user ratings to API based on sorted order"""
    try:
        # Convert sorted order to ratings (1 = highest rank)
        ratings_list = [
            {"artist_id": artist["artist_id"], "rank": idx + 1}
            for idx, artist in enumerate(sorted_artists)
        ]

        if not ratings_list:
            st.warning("Please arrange at least one artist before submitting!")
            return False

        payload = {"user_id": user_id, "ratings": ratings_list}

        response = requests.post(f"{API_BASE_URL}/ratings", json=payload)

        if response.status_code in [200, 201]:
            return True
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Failed to submit ratings: {error_detail}")
            return False
    except Exception as e:
        st.error(f"Error submitting ratings: {str(e)}")
        return False


def get_recommendations(user_id, top_k=10):
    """Get recommendations from API"""
    try:
        payload = {"user_id": user_id, "top_k": top_k}

        response = requests.post(f"{API_BASE_URL}/recommendations", json=payload)

        if response.status_code == 200:
            data = response.json()
            return data.get("recommendations", [])
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"Failed to get recommendations: {error_detail}")
            return None
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None


# Main App
st.markdown('<p class="title">🎵 Music Recommender</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Discover your next favorite artist</p>', unsafe_allow_html=True
)

# Login Section
if not st.session_state.logged_in:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### Welcome! Please enter your User ID")
        user_id_input = st.text_input(
            "User ID", placeholder="e.g., alice123", label_visibility="collapsed"
        )

        if st.button("Start Rating", use_container_width=True):
            if user_id_input.strip():
                st.session_state.user_id = user_id_input.strip()
                st.session_state.logged_in = True
                st.session_state.artists = fetch_artists()
                st.session_state.sorted_artists = st.session_state.artists.copy()
                st.rerun()
            else:
                st.warning("Please enter a valid User ID")

# Main Content (after login)
else:
    # Header with user info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 👤 Welcome, **{st.session_state.user_id}**")
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.sorted_artists = []
            st.session_state.recommendations = None
            st.session_state.current_rec_index = 0
            st.session_state.shown_recommendations = []
            st.rerun()

    st.markdown("---")

    # Load artists if not already loaded
    if not st.session_state.artists:
        st.session_state.artists = fetch_artists()
        st.session_state.sorted_artists = st.session_state.artists.copy()

    # Tabs for Rating and Recommendations
    tab1, tab2 = st.tabs(["🎸 Rank Artists", "✨ Get Recommendations"])

    with tab1:
        st.markdown("### Rank Your Favorite Artists")
        st.markdown("*Drag and drop to reorder - Top = Best (Rank 1)*")
        st.markdown("")

        if st.session_state.artists:
            # Create sortable list with images displayed inline
            artist_items = []

            for artist in st.session_state.sorted_artists:
                # Create a container for each artist with image
                artist_name = artist["artist_name"]
                image_path = get_artist_image(artist_name)

                # We'll use streamlit columns within sort_items
                artist_items.append(artist_name)

            # Display sortable items with custom rendering
            st.markdown(
                '<div style="background: #e6e6fa; padding: 10px; border-radius: 15px;">',
                unsafe_allow_html=True,
            )

            # Create columns for each artist to show image + name
            sorted_names = sort_items(
                artist_items, direction="vertical", key="artist_sorter"
            )

            # Update the sorted order based on user interaction
            new_order = []
            for sorted_name in sorted_names:
                for artist in st.session_state.artists:
                    if artist["artist_name"] == sorted_name:
                        new_order.append(artist)
                        break

            st.session_state.sorted_artists = new_order

            # Display sorted list with images
            st.markdown("#### Your Ranking:")
            for idx, artist in enumerate(st.session_state.sorted_artists):
                artist_name = artist["artist_name"]
                image_path = get_artist_image(artist_name)

                col1, col2 = st.columns([1, 5])
                with col1:
                    if image_path:
                        st.image(image_path, width=60)
                    else:
                        st.markdown("🎵")
                with col2:
                    st.markdown(
                        f"""
                        <div style="background: #b3d9ff; padding: 15px; border-radius: 10px; 
                             display: flex; align-items: center; justify-content: space-between;">
                            <span style="font-weight: 600; color: #000000; font-size: 1.1em;">
                                {artist_name}
                            </span>
                            <span class="rank-badge">#{idx + 1}</span>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.error("Failed to load artists. Please check if the API is running.")

        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📤 Submit Rankings", use_container_width=True):
                with st.spinner("Submitting your rankings..."):
                    if submit_ratings(
                        st.session_state.user_id, st.session_state.sorted_artists
                    ):
                        st.success("✅ Rankings submitted successfully!")
                        time.sleep(1)

    with tab2:
        st.markdown("### Your Personalized Recommendations")
        st.markdown("")

        # Show previously revealed recommendations
        if st.session_state.shown_recommendations:
            st.markdown("#### 🌟 Your Recommendations So Far:")
            for rec in st.session_state.shown_recommendations:
                rank = rec["rank"]
                artist_name = rec["artist_name"]
                score = rec["score"]

                # Try to load image
                image_path = get_artist_image(artist_name)

                col1, col2 = st.columns([1, 4])
                with col1:
                    if image_path:
                        st.image(image_path, width=80)
                    else:
                        st.markdown("🎵")

                with col2:
                    st.markdown(
                        f"""
                        <div class="recommendation-mini">
                            <span class="rank-badge">#{rank}</span>
                            <strong style="font-size: 1.1em;">{artist_name}</strong>
                            <span style="margin-left: auto; color: #7d6fa8; font-weight: 600;">
                                {score:.1%}
                            </span>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

        # Button to get or show next recommendation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state.recommendations is None:
                # First time - fetch recommendations
                if st.button("🎯 Get Recommendations", use_container_width=True):
                    with st.spinner("Generating recommendations..."):
                        recommendations = get_recommendations(
                            st.session_state.user_id, top_k=10
                        )
                        if recommendations:
                            st.session_state.recommendations = recommendations
                            st.session_state.current_rec_index = 0
                            st.session_state.shown_recommendations = []
                            st.rerun()
            else:
                # Show next recommendation
                if st.session_state.current_rec_index < len(
                    st.session_state.recommendations
                ):
                    if st.button("➡️ Next Recommendation", use_container_width=True):
                        # Add current recommendation to shown list
                        current_rec = st.session_state.recommendations[
                            st.session_state.current_rec_index
                        ]
                        st.session_state.shown_recommendations.append(current_rec)
                        st.session_state.current_rec_index += 1
                        st.rerun()
                else:
                    st.success("🎉 You've seen all recommendations!")
                    if st.button("🔄 Start Over", use_container_width=True):
                        st.session_state.recommendations = None
                        st.session_state.current_rec_index = 0
                        st.session_state.shown_recommendations = []
                        st.rerun()

        # Display current recommendation (if there is one to show)
        if (
            st.session_state.recommendations
            and st.session_state.current_rec_index
            < len(st.session_state.recommendations)
            and st.session_state.shown_recommendations
        ):
            # Show the most recently added recommendation in detail
            rec = st.session_state.shown_recommendations[-1]
            rank = rec["rank"]
            artist_name = rec["artist_name"]
            score = rec["score"]

            # Medal for top 3
            medal = ""
            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"
            else:
                medal = f"#{rank}"

            st.markdown("")
            st.markdown("#### 🎵 Current Recommendation:")

            # Try to load image
            image_path = get_artist_image(artist_name)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if image_path:
                    st.image(image_path, use_column_width=True)

                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <h2 style="color: #000000; margin: 0;">{medal} {artist_name}</h2>
                        <p style="color: #000000; margin-top: 15px; font-size: 1.2em;">
                            Match Score: <strong style="color: #7d6fa8; font-size: 1.3em;">{score:.1%}</strong>
                        </p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        elif not st.session_state.recommendations:
            st.info(
                "Submit your rankings first, then click 'Get Recommendations' to see personalized suggestions!"
            )
