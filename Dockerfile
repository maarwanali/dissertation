# 1. Choose the base computer
FROM python:3.11-slim

# 2. Set the working directory (creating a folder inside the container)
WORKDIR /app

# 3. Copy the ingredients list first
COPY requirements.txt .

# 4. Install the ingredients (libraries)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your Python scripts into the container
COPY . .

# 6. Set the default command (Azure will override this when running specific steps)
CMD ["python", "compare_static.py"]