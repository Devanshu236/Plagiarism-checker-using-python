# Plagiarism-checker-using-python

<p>Language Modeling
A language model is a statistical model that captures relevant linguistic features of the corpus on which it is trained. At a basic level, it should capture the frequency distribution of letters and words. A more advanced language model should capture syntactic and grammatical dependencies, such as agreement and inflection, and semantic properties, such as which words are likely to occur in a given context. Language models are typically used for two main tasks: scoring and generation. In scoring, the language model gives a probability score to a certain word occurring in a given context. Given the context machine learning is ____ , it should hopefully give a higher score to the completion fun than to the completion octopus , assuming it is trained on a representative sample of English text. In generation, the model samples from the learned distribution to generate fake but plausible-sounding text. Claude Shannon discussed language models in his seminal 1948 paper, and the ability to assign a probability to a sequence of words is a great achievement of information theory.
To apply language modeling to plagiarism detection, you can train a language model on a bunch of text that you think people may copy from. Let’s say you are currently teaching about the Gold Foil Experiment in a chemistry class. You could take several relevant Wikipedia articles, the top 10 Google search results for “gold foil experiment”, and perhaps student submission from the last several years (if you’re worried about hand-me-down papers) and dump them in a single text file. This aggregated dataset will be our training data that we use to build a language model, which captures the statistical features of the text. Once we have this language model, we can run student work through the language model to assign scores. A higher score means the work is more predictable from the training data, and represents a higher likelihood of plagiarism.
In a previous article, I discussed two different approaches for language modeling — N-gram models and RNNs. In that article, the goal was to generate fake text of a certain style. One of these methods, the N-gram model, turns out to be a simple but effective implementation for plagiarism detection.
N-gram Language Model
An N-gram language model scores words based on the preceding window of context. Although the N-gram model is not very sophisticated and fails to handle long-range dependencies and abstract semantic information, we can actually see this as a feature rather than a bug for this task. Other language models, such as those based on Recurrent Neural Networks or Transformers, are better at capturing long-range dependencies and higher levels of abstraction. For plagiarism, however, the emphasis is on copied sequences of words, not on similarities at an abstract level. A paraphrasing should not set off an alarm, but a direct copying should. I found that an N-gram window of 4 worked well, and it also aligns with the advice of many teachers not to use more than three words in a row from a source.
To implement an N-gram language model in Python, we can use the NLTK library (one option among many). The basic steps of training a language model are the following:
Read in and pre-process a training data file (e.g. remove punctuation, casing, and formatting). We would be left with something like this is an example sentence
Tokenize the training data (i.e. separate into individual words) and add padding at the beginning. This would leave us with ['<s>', '<s>', 'this', 'is', 'an', 'example', 'sentence'] .
Generate N-grams from the training data using the nltk.ngrams or nltk.everygrams methods. For an N-gram size of 3, this would give us something like [('<s>', '<s>', 'this'), ('<s>', 'this', 'is'), ('this', 'is', 'an'), ('is', 'an', 'example'), ('an', 'example', 'sentence')] . Note that everygrams would also give us the unigrams and bigrams, in addition to trigrams.
Fit a model using these N-grams. NLTK has various models that can be used, ranging from a basic MLE (Maximum Likelihood Estimator) to more advanced models like WittenBellInterpolated that use interpolation to deal with unseen N-grams.
Once we have the trained model, it supports various operations such as scoring a word given a context, or generating a word from the learned probability distribution. It is now time to evaluate some “student work”.
Read in an pre-process the testing data (the “student work”).
Tokenize the testing data.
For each word in the text, call model.score() on that word, with the previous N-1 words as the context argument.
This gives us a list of scores between 0 and 1, one per word, where a larger score represents a higher probability that the given words was plagiarized.
Visualization
Wouldn’t it be nice if we could visualize, at a glance, whether and how much a text was plagiarized? To do this, we can represent a student submission as a heatmap image where each pixel corresponds to the score of one word. This allows us to quickly gauge if plagiarism is likely, and which parts of a text were most likely to have been plagiarized. Visualizing information in this way is more useful that looking at an array of numerical scores or a summary statistic of all the scores.
I played around with various ways of visualizing this data, and I came up with the following method:
Display K words per line (I used K=8). This is the heatmap width in pixels. Then calculate the height (number of words in testing data divided by K).
Due to the small size of the dataset and the challenges of interpolation, there is some uncertainty in the assigned scores, so I applied Gaussian smoothing to the scores.
Reshape the array of scores into a rectangle of K columns and height rows. This requires adding zero padding to ensure the array is the correct size.
Use Plotly Heatmap to show the image using the colorscale of your choice.
Show the K words of text as a y-axis tick label next to the corresponding row of the heatmap for easy side-by-side comparison. Adjust the hover data so that each pixel shows its corresponding word on hover.
To test this approach, I trained the plagiarism detection model on the Wikipedia article for the Geiger-Marsden Experiment, also known as the Gold Foil experiment. For the “student work”, I compared two submissions, one which was copied with minor changes from Wikipedia, and one which came from a completely different source talking about the same topic.</p>
