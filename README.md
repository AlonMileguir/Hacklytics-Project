# MockMD: AI-Powered Clinical Training Simulator

## Development Instructions

## Local Development

To start the vector database server:

```sh
docker compose up
```

To start the FastAPI server:

```sh
cd app
uvicorn server:app --reload --port 8000
```

## Production Deployment

```sh
docker image build -t mock-md . --build-arg GEMINI_API_KEY=...
```

Run with:

```sh
docker run -p 80:80 mock-md
```

## Inspiration

Before Hacklytics, our group was interested in practical applications of vectorized databases and similarity search. As we were brainstorming, healthcare stood out as an area with realistic use cases. This led us to create a project that would help aspiring doctors and medical students practice clinical reasoning to prepare for their careers.

## What it does

MockMD is an AI-powered clinical training simulator that allows aspiring doctors to practice diagnosing patients with a wide range of conditions. Users begin by specifying what type of cases they want to explore (ex. "female over 50 with chest pain"). Then, MockMD uses a real case from PubMed to mimic the doctor-patient experience. The user is able to ask follow up questions, gather patient history, and reason through the diagnostic process. At the end of the session, MockMD provides personalized feedback on the responses the user gave to the patient. Additionally, MockMD has a feature to search and explore similar cases based on a user search.

## How we built it

We started by using Actian's Vector AI Database to store and retrieve clinical case data. Using a vectorized database allows for similarity search to match user requests to relevant cases rather than using keyword matching. Our database uses patient cases derived from PubMed's medical literature. We vectorized each of the documents to put into the Vector AI database. From there, we integrated Gemini's API to allow users to ask for cases that they want through a chatbot. After they submit to the chatbot, an agent performs vector similarity search to find cases that were similar to the one the user requested.

## What we learned

MockMD was the team’s first experience working with a vector database. Using this method of storage allowed us to learn about Approximate Nearest Neighbor algorithms and the process of vector encoding data. In addition we were able to make use of LLM API integration for a chatbot feature.
