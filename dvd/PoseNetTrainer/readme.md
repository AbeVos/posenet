## Benodigdheden voor PoseNetTrainer

Om PoseNetTrainer succesvol uit te voeren moet de volgende software aanwezig zijn op uw systeem:
- Python 3 of hoger
- CUDA 8.0 of hoger en de CUDA Toolkit

De volgende Python packages moeten geïnstalleerd zijn in uw omgeving:
- PyTorch (wordt momenteel alleen ondersteund op OSX en Linux)
- Numpy 1.12.1 of hoger
- Scipy
- Pygame
- OpenCV 3.2.0 of hoger
- Matplotlib

PoseNetTrainer bevat een aantal verschillende scripts. Hieronder volgt een lijst van de belangrijkste van deze scripts met een korte omschrijving van wat ze doen:
- posenet.py; bevat classes voor PoseNet en voor autoencoder lagen. Bevat ook enkele functies om modellen te laden en op te slaan.
- posenet_pretrainer.py; script om PoseNet als een reeks autoencoders te pretrainen op ongelabelde data.
- posenet_trainer.py; script om PoseNet te finetunes aan de hand van gepretrainde netwerken.
- NLabeler.py; grafisch programma dat videobestanden en numpy afbeeldingen kan laden zodat stukken uitgeknipt en gelabeld kunnen worden.
- generate_dataset.py; neemt een korte video op met de webcam en slaat deze op als een numpy array.

De map models bevat twee gepretrainde netwerken en het uiteindelijke PoseNet model.

