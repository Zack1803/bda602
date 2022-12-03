FROM ubuntu

RUN mkdir /scripts
RUN mkdir /results

COPY ./script.sh /scripts/script.sh
COPY ./baseball.sql /scripts/baseball.sql
COPY ./cmd.sql /scripts/cmd.sql

RUN chown 1000:1000 /scripts

RUN apt-get update && apt-get install -y default-mysql-client

RUN chmod +x /scripts/script.sh
CMD ./scripts/script.sh