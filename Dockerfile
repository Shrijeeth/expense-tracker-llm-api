FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./setup.py /code/setup.py

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install --upgrade setuptools

COPY ./ /code/

RUN pip install .

CMD ls

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]