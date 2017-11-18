# detecting-stance-in-tweets

Sometimes we want to analyze the post effects of any announcement or release likesome  movie   release,  GoT   episode   release,   government   policy   etc.   to   understand   thereaction . Stance detection can be formulated in different ways. In the context of thistask, we define stance detection to mean automatically determining from text whetherthe author is in favour of the given target, against the given target, or whether neitherinference is likely.
Automatically detecting stance has widespread applications in information retrieval,text   summarization,   and   textual   entailment.   In   fact,   one   can   argue   that   stancedetection can often bring complementary information to sentiment analysis, becausewe   often   care   about   the   author’s   evaluative   outlook   towards   specific   targets   andpropositions rather than simply about whether the speaker was angry or happy.

### Dependencies

  - hunspell
  - ntlk

[Download](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip) wordnet corpus and follow these [instructions](https://medium.com/@satorulogic/how-to-manually-download-a-nltk-corpus-f01569861da9)

```sh
$ pip install hunspell nltk
$ python
>> import nltk
>> nltk.download()
```

