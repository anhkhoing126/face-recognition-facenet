# Face Recognition using Facenet

## 1. Setup
Run ```create_dir_setup.sh``` in command line to create folders to store dataset, classifier.
```
chmod +x ./create_dir_setup.sh
./create_dir_setup.sh
```

Install the required dependencies form ```requiremnents.txt```
```
pip install -r requirements.txt (Python 2)
pip3 install -r requirements.txt (Python 3)
```

Download the pretrained model of Facenet and extract it into ```pretrained_model``` folder.
[Link](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view)

## 2. Run the files
```
1. Run collect_data.py to create new dataset
```
Enter your name to label the dataset and press ```q``` to stop collecting data. When press ```q``` the window won't collase and
you have to force stop it, I'm trying to fix it at the momemt.
```
2. Run align_rawdataset.py to align the faces from dataset
```
```
3. Run create_classifier.py to generate the classifier
```
```
4. Run recog_face_real_time.py to recognize face
```
Press ```q``` when you want to stop recognizing process.
