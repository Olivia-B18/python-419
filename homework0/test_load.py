import spotifysongs

# Load all tables as their own variable
globals().update(spotifysongs.load())

# Print the tables
print(songs.head())
print(genres.head())

