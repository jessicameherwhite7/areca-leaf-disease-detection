from flask import Flask,request,jsonify,render_template
import torch
import torchvision.models as models
import torch.nn as nn
import cv2
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import gdown

app=Flask(__name__)

if not os.path.exists("orb_gate_disease_model.pt"):
    url="https://drive.google.com/uc?id=1VJGDRJLgY-vV0FXDSgyT0kolkz6a95O1"
    gdown.download(url,"orb_gate_disease_model.pt",quiet=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("orb_areca_bank.pkl","rb") as f:
    areca_bank=pickle.load(f)

orb=cv2.ORB_create(2000)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

model=models.resnet18(weights=None)
model.fc=nn.Linear(model.fc.in_features,4)
model.load_state_dict(torch.load("orb_gate_disease_model.pt",map_location=device))
model=model.to(device)
model.eval()

transform=transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor()
])

disease_names=[
"Healthy",
"Bacterial Leaf Stripe",
"Leaf Blight",
"Yellow Leaf Disease"
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def predict():
    file=request.files["image"]
    path="temp.jpg"
    file.save(path)

    img_gray=cv2.imread(path,0)
    kp_input,des_input=orb.detectAndCompute(img_gray,None)

    similarity_scores=[]

    if des_input is not None:
        for des_db in areca_bank:
            if des_db is None:
                continue

            matches=bf.match(des_input,des_db)

            good_matches=0
            for m in matches:
                if m.distance < 30:
                    good_matches += 1

            similarity_scores.append(good_matches)

            if good_matches > 120:
                break

    best_score=max(similarity_scores) if similarity_scores else 0
    threshold=80

    if best_score<threshold:
        return jsonify({"result":"Not an areca leaf"})

    img=Image.open(path).convert("RGB")
    img_tensor=transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs=model(img_tensor)
        pred=torch.argmax(outputs,dim=1).item()

    return jsonify({"result":disease_names[pred]})

if __name__=="__main__":
    app.run(debug=True)

