def count_vowels(text):
    vowels = "aeiouAEIOU"
    count = 0
    for character in text:
        if character in vowels:
            count += 1
    return count