# riceclassification2.0
Projet de classification des types de riz utilisant PyTorch. Comprend le prétraitement des données, un modèle de réseau de neurones avec BCELoss, l'entraînement, l'évaluation et des tests unitaires pour valider le workflow et les fonctions utilitaires
```
riceclassification2.0/
│
├── requirements.txt          
├── README.md                 
├── .github/
│    
├── src/
│     ├── data_loader.py      
│     ├── model.py            
│     ├── train.py            
│     ├── utils.py            
│     └── workflow.py         
│
└── tests/
      ├── conftest.py         
      ├── test_data_loader.py 
      ├── test_model.py       
      ├── test_train.py      
      ├── test_workflow.py   
      └── test_utils.py      
      ````
pov: ce schema a ete realiser avec une intelligence artificielle

Le projet repose sur le fichier riceClassification.csv, comprenant plusieurs mesures géométriques des grains de riz :
Aire
Longueur de l’axe majeur
Longueur de l’axe mineur
Excentricité
Convexité
Périmètre
Roundness
Aspect ratio
Etc.
La colonne Class contient la catégorie du grain (problème de classification binaire).
On a obtenu une tres bonne performance. 
