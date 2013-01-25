from nltk.corpus import wordnet as wn


def monosemous():
    all_lemmas = list(wn.all_lemma_names())
    print "First ten lemmas: ", all_lemmas[:10]
    print "All lemmas: ", len(list(all_lemmas))

    # Only interested in single words
    single_word_lemmas = [x for x in all_lemmas if not '_' in x]
    print "First ten single word lemmas: ", single_word_lemmas[:10]
    print "Single word lemmas: ", len(single_word_lemmas)

    # Find monosemous
    monosemous = [x for x in single_word_lemmas if len(wn.synsets(x)) == 1]
    print monosemous[:10]
    senses = [(x, wn.synsets(x)[0]) for x in monosemous]
    noun_senses = [x for x in senses if x[1].pos == 'n']
    print "First ten monosemous nouns: ", noun_senses[:10]
    print "Number of monosemous nouns: ", len(noun_senses)
    return noun_senses

if __name__ == "__main__":
    m = monosemous()
    
