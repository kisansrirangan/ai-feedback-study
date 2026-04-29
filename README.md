# AI Feedback Study Prototype

Prototype for master's thesis: comparison of baseline prompting and criteria-enhanced prompting for AI-generated formative programming feedback.

## Requirements

* Python 3.10 or newer
* Internet connection
* OpenAI API key

## Setup

### 1. Open terminal and go to the project folder

```bash
cd path/to/Prototype
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

**Mac/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 4. Install required packages

```bash
python3 -m pip install -r requirements.txt
```

### 5. Add OpenAI API key

Create a file:

```bash
.streamlit/secrets.toml
```

Add:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

## Run the app

```bash
python3 -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Notes

* Responses are automatically stored in `responses.json`
* Pilot data is stored separately in `responses_pilot.json`
* The app uses the current prompt version defined in `app.py`

## If Streamlit is missing

If you get:

```text
No module named streamlit
```

activate the virtual environment again and reinstall requirements:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```