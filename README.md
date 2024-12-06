# 650-Reddit-SearchEngine

```markdown
# Running the Search Engine Locally

## Prerequisites
- Python 3.8 or higher installed on your system.
- Basic familiarity with Python and terminal commands.

## Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sameryassir/650-Reddit-SearchEngine.git
cd 650-Reddit-SearchEngine
```

### 2. Set Up the Virtual Environment
Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

Additionally, ensure **Uvicorn** is installed:
```bash
pip install uvicorn
```

### 4. Run the Search Engine
Start the FastAPI server:
```bash
uvicorn app:app --reload
```

### 5. Access the Web Interface
- Open a browser and navigate to: `http://127.0.0.1:8000`.
- Enter a search query to retrieve results.

### 6. Stopping the Server
To stop the server, press `Ctrl + C` in the terminal.
```
