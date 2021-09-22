import chardet

guess = chardet.detect(open('metatranscriptomes/salazar_profiles/salazar_metadata.csv', 'rb').read())['encoding']
print(guess)
