# aio-bot

## All in one bot

### Requirements

```sh
pip install -r requirements.txt
```
### Demo

```sh
python main.py
```
### List of features
1. [Add user to database](#add-user-to-database)
2. [Detect User at start](#detect-user-at-start)
3. [Detect Emotion](#detect-emotion)
4. [Add your question and answers to your datafile](#add-your-question-and-answers-to-your-datafile)

#### Add user to database
In Users folder add a photo of the user with image name as their name (only name.jpg format).
Copy the "Unknown.template" file and rename the new file by replacing "Unknown" with "name"

#### Detect User at start
If bot detects user and remembers him/her then he will great with their name else it will say "I am a chatbot"

#### Detect Emotion
Currently you have to have "feeling" in sentence for emotion detection.
It will reply your emotion.

#### Add your question and answers to your datafile
To save your pre answered question:
Write question in this format:
```
>learn mode ques {question in second person grammer}
```
Write answer in this format:
```
>learn mode ans {answer in second person grammer}
```

### To Do
- [x] Finish chat template
- [ ] Identify the user while starting the conversation(Not supported anymore)
- [X] Detect user's emotion (Currently detects only when user asks)
- [ ] Bot can remember the user name(Not supported anymore)
- [ ] Remember User's Details
- [X] Encrypts every user interaction using a complex crypto algo

