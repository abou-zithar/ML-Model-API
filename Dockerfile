FROM python:3.8
COPY ./mymain /app/mymain
COPY ./requirements.txt /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn","mymain.main:app","--host=0.0.0.0","--reload"]