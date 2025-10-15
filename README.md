Step 1: Clone the Repo and Navigate to RAG-Chatbot-Backend. Later create a virtual environment using below command and activate the virtual environment
  1.git clone URL
  2.python -m venv virtual_environment_name
  3.virtual_environment_name\Scripts\Activate.ps1 (powershell)

Step 2: Install all the libraries of python using below command
    pip install -r requirements.txt

Step 3: Run the below command so that the uvicorn is up and swagger page appears to interact with API's
    uvicorn api:apps --host 0.0.0.0 --port 8090


