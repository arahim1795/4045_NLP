# 4045_NLP

4045 Natural Language Processing

## Requirements

1. [Python](https://www.python.org/)
2. [Pipenv](https://github.com/pypa/pipenv)

## Setup

### Initial

1. Download and Install Python
2. Navigate to your Python installation directory and copy path
3. Open _cmd_ (administrator)
4. Input `cd <paste copied path>`
5. Input `py -x.x -m pip install --upgrade pip`
   1. Replace `x.x` with Python version number (i.e. downloaded: **3.7**.4, x.x: **3.7**)
6. Input `py -x.x -m pip install --upgrade setuptools`
   1. Replace `x.x` with Python version number (i.e. downloaded: **3.7**.4, x.x: **3.7**)
7. Input `pip install --user pipenv`
   1. Copy path presented after successfull installation: _looks like_ `C:\Users\<Username>\AppData\Roaming\Python<Version>\Scripts`
8. Add path into user's environment path

### Continued

1. Clone/Download project to a desired directory
2. Copy path to cloned/downloaded project directory after successful clone/download
3. Open _cmd_ (administrator)
4. Input `cd <path to cloned/downloaded project path>`
5. Input `pipenv install nltk`
6. Place dataset _(*.json)_ into Data folder

## Execution

1. To run, input `pipenv run python <file>.py`
