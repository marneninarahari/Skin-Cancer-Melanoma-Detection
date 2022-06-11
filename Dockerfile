# syntax=docker/dockerfile:1

# Set base image (host OS)
FROM python:3.9-slim-buster

# By default, listen on port 5000
EXPOSE 8000

# Set the working directory in the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the content of the local src directory to the working directory
COPY ./app /code/app


# Specify the command to run on container start

  # For debugging:
    # COMMAND FOR FLASK FRAMEWORK
#CMD ["python3", "-m" , "flask", "run", "--host=0.0.0.0"] 
    # COMMAND FOR fastAPI FRAMEWORK
CMD ["uvicorn", "app.main:app", "--reload", "--host=0.0.0.0"]

  # For production:
#CMD [ "gunicorn", "--workers=5", "--threads=5", "-b 0.0.0.0:5000", "app:server"]           # COMMAND FOR FLASK FRAMEWORK
#CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]    # COMMAND FOR fastAPI FRAMEWORK
