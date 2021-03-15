FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN pip3 install scikit-learn==0.22.2.post1
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install joblib

RUN pip3 install torch==1.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY ./app /app

COPY ./models /models

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]