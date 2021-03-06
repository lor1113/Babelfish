# Babelfish

Babelfish is a system to generate and utilize random languages, fundamentally based on English but with significant variations.
It is built on two main scrips: main.py, which generates new random languages, and translate.py, which translates English into any generated language. Generated language to English translation is not yet supported.

## Script Usage

### main.py

main.py is utilized by running the *language()* function. This will generate a language and place it in a new folder named with the language name inside a folder called "Languages" in the script directory. It will create this folder if absent.

The function takes two optional variables, *seed* and *state*. Seed accepts a 32bit integer seed to seed the Numpy Mersenne twister used for random number generation. State accepts an entire Mersenne twister state. Both allow for perfect reproducibility of language generation. Without either of these variables the generator is randomly seeded with np.random.seed() which uses OS-provided random entropy. *(if both seed and state are provided state overrides seed)*

It is possible to change the character set used from the english one, but it is finnicky so for now it should be considered fixed to it.

### translate.py

translate.py is utilized by running the *translate()* function. This simply takes a string input and returns the appropriate translation. **Additionally, the global variable** *name* **must be set to the name of the language to translate into.** The script assumes it is in the same directory as main.py and so looks for languages inside a directory called "Lanuages" inside its own directory. This is fairly easy to change by changing the *Folder* variable which points to the language folder.

## Language Generation

### Statistics Generation
The first step in language generation is to generate statistical characteristics which the rest of the script uses to generate the language. The generated characteristics are as follows:

#### Average Word Length
Length uses a lookup table of 20000 average word lengths, from 0 to 20, and the number of characters required at said word length to achieve the same wordspace as english (in which wordspace is simply the length of the character set to the power of the average word length, as an attempt to express how much "space" the language has for words). The script finds which number of characters entry in the array most closely matches the number of characters of the character set being used, and uses it as the average word length. It is further randomised by +-25%.

**Note: Average word length means the average of all the words in the language counted once, not the average of words in a text or corpus**

#### Average Word Length Variance
Variance uses a simple inverse relation of being 30/average word length. This means longer average lenths have lower variation and vice versa. This was to ensure that since words can only be integers long, low average word length languages still have some word length variation, otherwise any variation would have been swallowed into rounding.

#### Markov Generation
The script uses "Markovs" to generate words for the language. There are three "Markovs" used, a simple letter frequency distribution, a Markov set from character to character, and a 2-deep Markov set which gives probabilities based on the preceding 2 characters. Letter distibution is based on the Cocho/Beta function as described here (https://tinyurl.com/v6aabes). The base values used are 0.15 and 1.75. Both the 2-deep and "normal" markov are generated using this function with those base values, each time randomizing them both by +-50%. The letter distribution is then generated by generating a long letter sequence with the 2-long markov (seeding it with a pure random choice) and counting.

#### Shiftset Generation
Shiftsets represent how the language shuffles around words in English to simulate a word order. Shiftsets are pairs of sequences n long, formed of digits 0-9,\*,and letters. The sequences are shuffled versions of one another, with identical consitituents. Currently shiftsets are generated for lengths 2-9, although the uppper limit can be increased (lower than 2 is not feasible however).

Shiftsets represent commands to shuffle words in certain ways. First, words are tagged based on their type (ie: noun, adverb, adjective, etc) with these 10 tags mapped to the digits 0-9. In the language as a whole, some of these tags are also randomly grouped up, with the group assigned a letter. Then, sentences are split into chunks matching the length of the current shiftset being used. Then each chunk is converted into the sequence format. * in this scenario is a wildcard accepting any tag. Then, if a matching entry in the synset is found (Example: sequence is 5299 and there is a synset entry of 52\*9) the sequence is shifted around to match the sequence's shuffled pair in the shiftset. This is done for each shiftset separately, in descending order of length, with each shiftset accepting the previous one's shuffled output as an input.

Shiftsets are generated by randomly putting some of the 10 tags into a random number of groups, which are sequentially assigned alphabetical letters. The shiftset sequences are then randomly selected from the combined set of 0-9+\*+any groups with no weighting

### WordGen
The script generates words using three main characteristics: mean, variance, and the Markovs.

First a length is generated based on the mean and variance. Then a letter is picked based on the letter frequency table. A second letter is picked based off the 1-deep Markov. Further letters are chosen by the 2-deep markov until the word length is reached.

### Translation

To create the actual mapping between the words of English and the actual generated language, synsets are used. Synsets are part of NLTK, the Natural Language ToolKit, which represents the various meanings words can take. For each English word, a new word is generated, to which roughly half of the original word's meanings will be assigned. Additionally if there are any "leftover" meanings it may randomly acquire one of those. Once all english words have been parsed in this manner, completely new words are generated from the pool of leftover meanings, with the system attempting to generate words with multiple similar/related meanings.

## Config

To be added when the system actually has significant config capability
