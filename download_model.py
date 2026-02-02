from roboflow import Roboflow

# Use your actual API key
rf = Roboflow(api_key="fcCYBiPPncENLdquyj7F")


# Download license plate detection model
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(4)
dataset = version.download("yolov8")

print("Model downloaded successfully!")
print("Dataset location:", dataset.location)