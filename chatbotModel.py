#manipulation of DATASETS***********************************
#(NOTE:To import a dataset that is stored in your local computer to your IDE, you can use the file path of the dataset in your code to read it)
#example: file_path = "/../../../dataset.csv"/dataset = pd.read_csv(file_path)
import pandas as pd
import pandas as pd
from flask import Flask, jsonify, request

d= pd.read_csv("training_data.csv",sep=',')
g=pd.read_csv("final_disease_symptom_data.csv",sep=',')
f=pd.read_csv("dis_sym_dataset_comb.csv",sep=',')
d=d.rename(columns={"prognosis": "illness"})
d = d.drop('Unnamed: 133', axis=1)
f=f.rename(columns={"label_dis": "illness"})
g=g.rename(columns={"diseases": "illness"})
g= g.drop('Unnamed: 0', axis=1)
#Phase2:combining the datasets and remove the duplicates comluns
b = pd.concat([d, f, g])
illness_col = b.pop("illness")

b.insert(0, "illness", illness_col)
#replacing Nan values to 0 
b.fillna(0, inplace=True)
b = b[~b['illness'].isin(['Varicose veins','Diabetes','Hepatitis','Hepatitis A','Hepatitis B','Hepatitis C','Anxiety','Asthma','Arthritis','Anaemia','Chickenpox','Common cold','Colitis','Epilepsy','Gastroenteritis','Hypoglycemia','Hypertension','Influenza','Lymphoma','Migraine','Myocardial Infarction (Heart Attack)','Neoplasm','Pneumonia','Osteoporosis','Osteomyelitis','Schizophrenia','Sickle-cell anemia','Sepsis'])]
q=['Alcohol Abuse and Alcoholism','anxiety  state','Autism','bipolar  disorder','depressive disorder','Obsessive Compulsive Disorder','GERD', 'Typhoid','Amblyopia', 'Aniseikonia', 'Anisometropia', 'Chalazion', 'Corneal Abrasion', 'Glaucoma', 'Keratoconus', 'Leprosy', 'Lice', 'Nasal Polyps', 'migraine  disorders', 'epilepsy', 'Myasthenia gravis', 'Narcolepsy', 'Autism', 'Cerebral palsy', 'Insomnia', 'Multiple sclerosis','Coronary Heart Disease',
 'failure  heart',
 'failure  heart congestive',
 'Hypertension ',
 'myocardial  infarction',
 'cardiomyopathy',
 'mitral  valve insufficiency',
 'Pericarditis',
 'pericardial effusion body substance',
 'endocarditis',
 'Pulmonary embolism',
 'deep  vein thrombosis',
 'Acne',
 'Eczema',
 'Psoriasis',
 'Chicken pox',
 'melanoma',
 'Warts',
 'Yaws',
 'Smallpox',
 'cellulitis',
 'Impetigo',
 'Scabies',
 "Alzheimer's  disease",
 'Brain Tumour',
 'parkinson  disease',
 '(vertigo) Paroymsal  Positional Vertigo',
 'Stroke',
 'Paralysis (brain hemorrhage)',
 'psychotic  disorder',
 'suicide  attempt',
 'personality  disorder',
 'paranoia',
 'manic  disorder',
 'schizophrenia',
 'ulcer  peptic',
 'Colorectal Cancer',
 'colitis',
 'Irritable bowel syndrome',
 'Celiacs disease',
 'pancreatitis',
 'cirrhosis',
 'gastroenteritis',
 'diverticulitis',
 'diverticulosis',
 'asthma',
 'pneumonia',
 'Lung cancer',
 'Tuberculosis',
 'Pulmonary embolism',
 'respiratory  failure',
 'Bronchitis',
 'emphysema  pulmonary',
 'Common Cold',
 'Coronavirus disease 2019 (COVID-19)',
 'upper  respiratory infection',
 'carcinoma of lung',
 'Breast Cancer / Carcinoma',
 'Lung cancer',
 'Colorectal Cancer',
 'Oral Cancer',
 'AIDS',
 'Chicken pox',
 'Dengue',
 'hepatitis A',
 'hepatitis  B',
 'hepatitis  C',
 'Malaria','influenza']
b= b[b['illness'].isin(q)]
b= b.loc[:, (b != 0).any(axis=0)]

#************************************************
df=b
data=df
# Create a list of unique illnesses
illnesses = data['illness'].unique()

# Create a dictionary to store the symptoms for each illness
illness_symptoms = {}
for illness in illnesses:
    symptoms = data[data['illness']==illness].iloc[:,1:].columns[data[data['illness']==illness].iloc[:,1:].sum() > 0].tolist()
    illness_symptoms[illness] = symptoms

# Create a new dataframe with unique illnesses and symptoms
new_data = pd.DataFrame(columns=['illness'] + list(data.columns[1:]))
for illness in illnesses:
    symptoms = illness_symptoms[illness]
    new_row = [illness] + [1 if symp in symptoms else 0 for symp in data.columns[1:]]
    new_data.loc[len(new_data)] = new_row
data=new_data
df=new_data

b = b.rename(columns={'red': 'red eyes'})
df = df.rename(columns={'red': 'red eyes'})
b = b.rename(columns={'suicidal': 'suicidal thoughts'})
df = df.rename(columns={'suicidal': 'suicidal thoughts'})

#*******************************************
# Filter the DataFrame to only include rows for the specified illness
def symp_list(illnes):
   illness_df = df[df['illness'] == illnes]

# Filter the DataFrame to only include columns with a value of 1 in the specified rows
   symptoms = list(illness_df.columns[illness_df.iloc[0,:] == 1])

# Print the list of symptoms
   return symptoms

#*******************************************
#MODEL************************************************
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#Random Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y = b['illness'].values
X = b.drop('illness', axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf2 = RandomForestClassifier(n_estimators=100, max_depth=120)
rf2.fit(X_train, y_train)

train_accuracy = rf2.score(X_train, y_train)
test_accuracy = rf2.score(X_test, y_test)


from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = rf2.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

#********************************************************



#VECTOR FUNCTION ************************************
m = b.drop('illness', axis=1)  # features
symp=[]
symp=m.columns
def modelCall(L):
 
 input_vector = [1 if symptom.strip() in L else 0 for symptom in symp]
 input_vector
 import numpy as np

 input_array = np.array(input_vector).reshape(1, -1)
 input_array.shape
 predictions = rf2.predict(input_array)
 return predictions[0]
#**********************************************************




# Define cluster symptoms******************************************************
from pickle import TRUE
import random
clusters = {
    1: {
        "key_symptoms": ['1 Have you experienced symptoms such as fever, fatigue, cough'],
        "illnesses": {
            "Malaria": symp_list("Malaria"),
            "Dengue": symp_list("Dengue"),
            "Tuberculosis": symp_list("Tuberculosis"),
            "Coronavirus disease 2019 (COVID-19)": symp_list("Coronavirus disease 2019 (COVID-19)"),
            "Influenza": symp_list("influenza"),
            "Common cold": symp_list("Common Cold"),
            "AIDS": symp_list("AIDS")
        }
    },
    2: {
        "key_symptoms": ['2 Do you have any skin-related problems or conditions, such as rashes, acne, warts, or infections?'],
        "illnesses": {
            "Chickenpox": symp_list("Chicken pox"),
            "Dengue": symp_list("Dengue"),
            "Impetigo": symp_list("Impetigo"),
            "Psoriasis": symp_list("Psoriasis"),
            "Scabies": symp_list("Scabies"),
            "Smallpox": symp_list("Smallpox"),
            "Warts": symp_list("Warts"),
            "Yaws": symp_list("Yaws"),
            "Cellulitis": symp_list("cellulitis"),
            "Lice": symp_list("Lice")
        }
    },
    3: {
       "key_symptoms": ['3 Do you have any vision problems or discomfort in your eyes?'],
    "illnesses": {
        "Amblyopia": symp_list("Amblyopia"),
        "Aniseikonia": symp_list("Aniseikonia"),
        "Anisometropia": symp_list("Anisometropia"),
        "Chalazion": symp_list("Chalazion"),
        "Corneal Abrasion": symp_list("Corneal Abrasion"),
        "Glaucoma": symp_list("Glaucoma"),
        "Keratoconus": symp_list("Keratoconus"),
    }

    },
    4: {
        "key_symptoms": ["4 Do you have any neurological symptoms such as headaches, seizures, muscle weakness, difficulty sleeping, or problems with movement?"],
    "illnesses": {
        "Alzheimer's Disease": symp_list("Alzheimer's  disease"),
        "Migraine": symp_list('migraine  disorders'),
        "Epilepsy": symp_list('epilepsy'),
        "Myasthenia Gravis": symp_list('Myasthenia gravis'),
        "Narcolepsy": symp_list('Narcolepsy'),
        "Brain Tumor": symp_list('Brain Tumour'),
        "Autism Spectrum Disorder": symp_list('Autism'),
        "Cerebral Palsy": symp_list('Cerebral palsy'),
        "Insomnia": symp_list('Insomnia'),
        "Parkinson's Disease": symp_list('parkinson  disease'),
        "Multiple Sclerosis": symp_list('Multiple sclerosis'),
        "Vertigo": symp_list('(vertigo) Paroymsal  Positional Vertigo'),
        "Stroke": symp_list('Stroke'),
        "Paralysis (Brain Hemorrhage)": symp_list('Paralysis (brain hemorrhage)')
    }
    },
    5: {
          "key_symptoms": ['5 Have you experienced persistent changes in mood or behavior that affect your daily life, such as feeling excessively worried or anxious, having intense mood swings or having thoughts of self-harm?'],
    "illnesses": {
        "Alcohol Abuse and Alcoholism": symp_list('Alcohol Abuse and Alcoholism'),
        "Anxiety state": symp_list('anxiety  state'),
        "Autism": symp_list('Autism'),
        "bipolar disorder": symp_list('bipolar  disorder'),
        "depressive disorder": symp_list('depressive disorder'),
        "Obsessive Compulsive Disorder": symp_list('Obsessive Compulsive Disorder'),
        "psychotic disorder": symp_list('psychotic  disorder'),
        "suicide attempt": symp_list('suicide  attempt'),
        "personality disorder": symp_list('personality  disorder'),
        "paranoia": symp_list('paranoia'),
        "manic disorder": symp_list('manic  disorder'),
        "schizophrenia": symp_list('schizophrenia')
    }
    },
    6: {
          "key_symptoms": ['6 Have you experienced any persistent gastrointestinal symptoms such as abdominal pain, bloating, nausea, vomiting, diarrhea, or constipation?'],

         "illnesses": {
        "GERD": symp_list('GERD'),
        "Peptic ulcer disease": symp_list('ulcer  peptic'),
        "Colorectal Cancer": symp_list('Colorectal Cancer'),
        "colitis": symp_list('colitis'),
        "Irritable bowel syndrome": symp_list('Irritable bowel syndrome'),
        "Typhoid": symp_list('Typhoid'),
        "Celiacs disease": symp_list('Celiacs disease'),
        "pancreatitis": symp_list('pancreatitis'),
        "cirrhosis": symp_list('cirrhosis'),
        "Gastroenteritis": symp_list('gastroenteritis'),
        "diverticulitis": symp_list('diverticulitis'),
        "diverticulosis": symp_list('diverticulosis')
   
    }
    },
    7: {
      "key_symptoms": ['7 Have you been experiencing any difficulty breathing or chest discomfort?'],
    "illnesses": {
        "Asthma": symp_list("asthma"),
        "Pneumonia": symp_list("pneumonia"),
        "Lung cancer": symp_list("Lung cancer"),
        "Tuberculosis": symp_list("Tuberculosis"),
        "Pulmonary embolism": symp_list("Pulmonary embolism"),
        "Respiratory failure": symp_list("respiratory  failure"),
        "Bronchitis": symp_list("Bronchitis"),
        "Emphysema": symp_list("emphysema  pulmonary"),
        "Upper respiratory infection": symp_list("upper  respiratory infection"),
        "Carcinoma of lung": symp_list("carcinoma of lung"),
    }
    },
    8: {
         "key_symptoms": ['8 Have you noticed any unusual changes in your breast/lung/colon/oral area, such as lumps, lesions, or discoloration?'],
    "illnesses": {
        "Breast Cancer": symp_list("Breast Cancer / Carcinoma"),
        "Lung cancer": symp_list("Lung cancer"),
        "Colorectal Cancer": symp_list("Colorectal Cancer"),
        "Oral cancer": symp_list("Oral Cancer"),
        "Melanoma": symp_list("melanoma"),
    }
    },
    9: {
          "key_symptoms": ["9 Have you experienced any symptoms related to your heart or cardiovascular system, such as chest pain, shortness of breath, fatigue, palpitations, or swelling in your legs?"],
    "illnesses": {
        "Heart failure congestive": symp_list("failure  heart congestive"),
        "Heart failure": symp_list("failure  heart"),
        "Hypertension": symp_list("Hypertension "),
        "Heartattack": symp_list("myocardial  infarction"),
        "Cardiomyopathy": symp_list("cardiomyopathy"),
        "Mitral valve prolapse": symp_list("mitral  valve insufficiency"),
        "Pericarditis": symp_list("Pericarditis"),
        "Endocarditis": symp_list("endocarditis"),
        "Pulmonary embolism": symp_list("Pulmonary embolism"),
        "Deep vein thrombos": symp_list("deep  vein thrombosis"),
    }
    },
}
#*******************************************

#chatbot**********************************************
def CHATBOT():
    # Select a random cluster
 answer = 'no'
 v=0
 asked_clusters = []
 while answer == 'no' and len(asked_clusters) < len(clusters):
    cluster = random.choice(list(clusters.values()))
    if cluster not in asked_clusters:
        @app.route('/predict', methods=['POST'])
        data = request.json
        input_df = pd.DataFrame([data])
        answer = input_df
        if answer == 'no':
            asked_clusters.append(cluster)
    if len(asked_clusters) == len(clusters):
        print('EXIT')

        

 if len(asked_clusters) < len(clusters):
    d=0
    num_illnesses = len(cluster["illnesses"]) 
    l=0  
    already=[]
    # Select a random illness
    while d!=1 and l<=num_illnesses:
        illness_name, illness_symptoms = random.choice(list(cluster["illnesses"].items()))
        symptom_count = len(illness_symptoms)
        yes_count = 0  
        no_counts=0
        while (yes_count / symptom_count <0.5) and (no_counts / symptom_count <0.5) :
            for symptom in illness_symptoms:
                if symptom not in already:
                    @app.route('/predict', methods=['POST'])
                    data = request.json
                    input_df = pd.DataFrame([data])
                    answer = input_df
                    already.append(symptom)
                    if answer.lower() == "yes":
                        yes_count=yes_count+1
                    if answer.lower() == "no":
                        no_counts=no_counts+1   
                    
                    if (yes_count/symptom_count) >=0.5:
                      w=modelCall(illness_symptoms)
                      if w==illness_name.lower():
                         print(w)
                         d=1
                         break
                      else:  
                        return jsonify( {'you may have '+illness_name+' go see a doctor'})
                        d=1
                        break  
                        
                    if (no_counts/symptom_count) >=0.5 :
                        
                        l=l+1
                        break
                else:
                    # Symptom already asked before, skip
                    
                    v=5
                    continue

        if d==1:
            # Diagnosis found, break out of while loop
            break

    if l>num_illnesses:
        print('We were unable to diagnose your condition.')


CHATBOT()
