FROM ubuntu

#RUN mkdir /results

RUN chown 1000:1000 ./

RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     libmysqlclient-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*


COPY ./script.sh /script.sh
COPY ./baseball.sql /baseball.sql
COPY ./project.sql /project.sql
COPY ./project.py /project.py
COPY ./requirements.txt /requirements.txt

RUN pip3 install --compile --no-cache-dir -r /requirements.txt
RUN pip install Jinja2
RUN apt-get update && apt-get install -y default-mysql-client


RUN chmod +x /script.sh
CMD ./script.sh

RUN python3 /project.py
