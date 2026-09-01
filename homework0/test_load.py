import spotifysongs

# Load all tables as their own variable
globals().update(spotifysongs.load())

# Print the tables
print("The length of the 'songs' DataFrame is: ", len(songs))
print(songs.head())
print()
print("The lenght of the 'genres' DataFrame is: ", len(genres))
print(genres.head())

