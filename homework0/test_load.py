import spotifysongs

# Load all tables as their own variable
globals().update(spotifysongs.load())

# Separate song info cols from genre cols, which are many-hot encoded
genre_cols = [col for col in songs.columns if col.startswith("genre_")]
time_sig_cols = [col for col in songs.columns if col.startswith("time_signature_")]
key_cols = [col for col in songs.columns if col.startswith("key_")]
info_cols = [col for col in songs.columns if not col.startswith("genre_") and
             not col.startswith("time_signature_") and
             not col.startswith("key_")
             ]

# Get dictionary of data types and columns (dtype, cols_with_that_dtype)
info_cols_dtypes = list(set([songs[info_col].dtype for info_col in info_cols]))
dict_dtypes = dict()
for key in info_cols_dtypes:
    dict_dtypes[key] = []
for col in info_cols:
    dict_dtypes[songs[col].dtype].append(col)

# Print info on the "songs" DataFrame
print()
print("========== The 'songs' DataFrame ===========")
print()

print("Number of songs: ", len(songs))
print("Number of total columns: ", len(songs.columns))

print()
print("========== SONG INFO ==========")
print()

print("Number of info columns: ", len(info_cols))
print("Info column names: ", songs[info_cols].columns)

print()
print("Data types of info columns:")

for key, value in dict_dtypes.items():
    print(key, " : ", value)

print()
print("========= TIME SIGNATURE COLUMNS ==========")
print()

print("Number of time signature columns: ", len(time_sig_cols))
print("Time signature column names: ", songs[time_sig_cols].columns)

print()
print("Counts:")
print(songs[time_sig_cols].sum().sort_values(ascending=False))

print()
print("========= SONG KEY COLUMNS ==========")
print()

print("Number of song key columns: ", len(key_cols))
print("Song key column names: ", songs[key_cols].columns)

print()
print("Counts:")
print(songs[key_cols].sum().sort_values(ascending=False))

print()
print("========== GENRE COLUMNS ==========")
print()

print("Number of one-hot genre columns: ", len(genre_cols))
# print("Genre column names: ", songs[genre_cols].columns)

print()
print("Genre columns statstics:")
print(songs[genre_cols].head())

print()
print("Genres per song:")
print(songs[genre_cols].sum(axis=1).describe())

print()
print("Ten most common genres:")
print(songs[genre_cols].sum().sort_values(ascending=False).head(10))

print()
print("Ten least common genres:")
print(songs[genre_cols].sum().sort_values(ascending=True).head(10))

print()
