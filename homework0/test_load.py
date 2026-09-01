import spotifysongs

# Load all tables as their own variable
globals().update(spotifysongs.load())

# Print the tables
print("The 'songs' DataFrame:")
print("Number of songs: ", len(songs))
print("Columns: ", songs.columns) 
print(songs.head())
print()

print("The 'genres' DataFrame:")
print("Number of song-genre pairs: ", len(genres))
print("Columns: ", genres.columns)
print(genres.head())
print()

