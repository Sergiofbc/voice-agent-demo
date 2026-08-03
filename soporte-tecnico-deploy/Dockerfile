FROM dailyco/pipecat-base:latest

COPY ./requirements.txt requirements.txt

# Uncomment this if you wish to print a summary of the features available in the base image.
# ENV PCC_LOG_FEATURES_SUMMARY=true

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./bot.py bot.py
