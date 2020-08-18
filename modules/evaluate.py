import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.sequence import pad_sequences
from modules.utils import progressBar

# generate a description for an image greedy way
def generate_desc(model, photo_fe, wordtoix, ixtoword, max_length, inference= False, start_token='<startseq>', end_token='<endseq>'):
    # seed the generation process
    in_text = start_token
    # iterate over the whole length of the sequence
    # generate one word at each iteratoin of the loop
    # appends the new word to a list and makes the whole sentence
    for i in range(max_length):
        # integer encode input sequence
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        # pad input
        photo_fe = photo_fe.reshape((1,2048))
        sequence = pad_sequences([sequence], maxlen=max_length).reshape((1,max_length))
        # predict next word
        yhat = model.predict([photo_fe,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = ixtoword[yhat]
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next v
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == end_token:
            break

    if inference == True:
        in_text = in_text.split()
        if len(in_text) == 34:
            in_text = in_text[1:] #if endseq hasn't appeared
        else:
            in_text = in_text[1:-1]
        in_text = ' '.join(in_text)

    return in_text


def beam_search_pred(model, pic_fe, wordtoix, ixtoword, max_length, K_beams = 3, log = False, start_token= '<startseq>', end_token='<endseq>'):
    start = [wordtoix[start_token]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence  = pad_sequences([s[0]], maxlen=max_length).reshape((1,max_length)) #sequence of most probable words
                                                                                         # based on the previous steps
            preds = model.predict([pic_fe.reshape(1,2048), sequence])
            word_preds = np.argsort(preds[0])[-K_beams:] # sort predictions based on the probability, then take the last
                                                         # K_beams items. words with the most probs
            # Getting the top <K_beams>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:

                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                if log:
                    prob += np.log(preds[0][w]) # assign a probability to each K words4
                else:
                    prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])

        # Getting the top words
        start_word = start_word[-K_beams:]

    start_word = start_word[-1][0]
    captions_ = [ixtoword[i] for i in start_word]

    final_caption = []

    for i in captions_:
        if i != end_token:
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption



def normalize_ref_and_pred(actual, predicted):
    # normalize references
    ref = {}
    for i, line in enumerate(actual):
        sentences = []
        for sentence in line:
            sentence = [word for word in sentence if word != '<startseq>']
            sentence = [word for word in sentence if word != '<endseq>']
            sentences.append(' '.join(sentence))
            # store
        ref[i] = sentences

    # normalize predicted

    new_pred = {}
    for i, line in enumerate(predicted):
        pred = " ".join(line).replace("<startseq>", "").replace("<endseq>", "")
        new_pred[i] = pred

    return ref, new_pred






def evaluate_model(model, descriptions, photos_fe, wordtoix, ixtoword, max_length, K_beams= 3, log=False):
    actual, predicted, image_ids = list(), list(), list()
    # step over the whole set
    i=0
    for key, desc_list in descriptions.items():
        # generate description
        i+=1
        progressBar(i, len(descriptions), bar_length=20,job='Evaluating')
        if K_beams == 1:
            yhat = generate_desc(model, photos_fe[key], wordtoix, ixtoword, max_length)
        else:
            yhat=beam_search_pred(model, photos_fe[key], wordtoix,ixtoword, max_length, K_beams = K_beams,log=log)

        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        image_ids.append(key)


    # convert results to a more readable format
    new_ref, new_pred = normalize_ref_and_pred(actual, predicted)

    # print results
    print("\n\nActual captions :")
    for key, value in new_ref.items():
        if key == 128:
            for i, val in enumerate(value):
                print("{} - {}".format(i, val))
            print("Predicted caption - {}".format(new_pred[key]))


    # calculate BLEU score
    b1=corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2=corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3=corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    b4=corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('\n')
    print('BLEU-1: %f' % b1)
    print('BLEU-2: %f' % b2)
    print('BLEU-3: %f' % b3)
    print('BLEU-4: %f' % b4)
    return actual, predicted, image_ids
