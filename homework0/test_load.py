import spotifysongs

# Load all tables as their own variable
globals().update(spotifysongs.load())

# Genres are one-hot encoded, so they occupy one column each
genre_cols = [col for col in songs.columns if col.startswith("genre_")]
info_cols = [col for col in songs.columns if not col.startswith("genre_")]

# Print info on the "songs" DataFrame
print()
print("===== The 'songs' DataFrame =====")

print()
print("Number of songs: ", len(songs))
print("Number of total columns: ", len(songs.columns))

print()
print("Number of song info columns: ", len(info_cols))
print("Song info columns: ", songs[info_cols].columns)

print()
print("Number of one-hot genre columns: ", len(genre_cols))
print("Genre columns: ", songs[genre_cols].columns)

print()
print("Song info cols:")
print(songs[info_cols].head())

print()
print("Genre cols:")
print(songs[genre_cols].head())

print()
print("Genres per song:")
print(songs[genre_cols].sum(axis=1).describe())

print()
print("Ten most common genres:")
print(songs[genre_cols].sum().sort_values(ascending=False).head(10))

print()
