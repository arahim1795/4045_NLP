# 4045_NLP

4045 Natural Language Processing

---

## Installing Dependencies

### Manual Installation

1. [OpenJDK 8](https://adoptopenjdk.net/) - [GNU General Public License 2.0](https://openjdk.java.net/legal/gplv2+ce.html)
2. [Python 3.7.4](https://www.python.org/downloads/) - [PSF Licence](https://docs.python.org/3/license.html)

### Manual Download

1. [Stanford CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/index.html#download) - [GNU General Public License 3.0](http://www.gnu.org/licenses/gpl-3.0.html)

   1. unzip zipped file
   2. move `stanford-corenlp-full-2018-10-05` to _Desktop_

2. [Stanford CoreNLP NER](https://nlp.stanford.edu/software/CRF-NER.html) - [GNU General Public License 3.0](http://www.gnu.org/licenses/gpl-3.0.html)

   1. unzip zipped file
   2. enter directory `stanford-ner-2018-16`
   3. copy `stanford-ner.jar` to _Desktop_
   4. enter directory `classifiers`
   5. copy `english.all.3class.distsim.crf.ser.gz` to _Desktop_

### Required Python Libraries

1. [MatPlotLib](https://matplotlib.org/) - [PSF Licence](https://docs.python.org/3/license.html)
2. [NumPy](https://numpy.org/) - [NumPy License](https://numpy.org/license.html)
3. [nltk](https://www.nltk.org) - [Apache License Version 2.0](www.apache.org/licenses/LICENSE-2.0)
4. [Spacy](https://stanfordnlp.github.io/stanfordnlp/index.html#get-started) - [MIT Licence](https://github.com/explosion/spaCy/blob/master/LICENSE)
5. [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/index.html#get-started) - [GNU General Public License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### Installation Steps

1. Once Python has been installed, input in _cmd_ : `pip install -r requirements_win.txt`
2. From Desktop, copy `stanford-corenlp-full-2018-10-05` into project folder _server_
3. From Desktop, copy `stanford-ner.jar` and `english.all.3class.distsim.crf.ser.gz` into project folder _lib_
4. Place `reviewSamples20.json` and `reviewSelected100.json` into project folder _data_

---

## Launch Project

1. Input in _cmd_ : python main.py
2. There will a prompt to install `en_ewt` and `en_gum` models, enter `y` to install
3. Await till program ends
4. Outputs accessible in project folder _out_
