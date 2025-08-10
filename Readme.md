# Insurance AI Agent

## Setup Instructions

### 1. Create a Virtual Environment

Use **Python 3.12 or below** (Python 3.13 might face issue while installing some dependencies).

```bash
python3 -m venv env_name
source env_name/bin/activate  
```

### 2. Install PortAudio

This is required before installing `pyaudio`, otherwise installation may fail.

#### On macOS:
```bash
brew update
brew install portaudio
```

#### On Linux:
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev
```

### 3. Install Python Dependencies

Make sure your virtual environment is activated:

```bash
pip install -r requirements.txt
```

## Running the Server

After installing all dependencies, start the server using:

```bash
uvicorn server:app --reload
```

It will produce a local development URL in the terminal (e.g., `http://127.0.0.1:8000`), which can be used to test the application.
# Conversational-bot
