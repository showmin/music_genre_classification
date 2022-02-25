# Music Genre Classification

A Python practice project for music genre classification. As the definition given by [GTZAN dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification), 10 classes (blues, classical, country, disco, hiphop, jazz, metal, pop, teggae, rock) are presented in the dataset and each has 100 sound clips.

---

**Note** that there is a broken clip in jazz (jazz.00054.wav) and you'd better remove it from your dataset to make the process go well.

---

In this project, I focused on training different machine learning models by applying mfcc coefficients of the training set and evaluating the performance.

# Get Started

1. Edit ```preproc.py``` to assign dataset path and output json path.

   ```python
   DATA_PATH = "Data/genres_original"
   JSON_PATH = "predata.json"
   ```

2. Generate the mfcc json file.  
   ```python preproc.py```

3. Run the program by a specific model.  
   ```python main.py --model crnn```  
   If you have different json path, remember to edit line 188 to 192 in ```main.py```.

4. Have fun.

   

# Evaluation Results

| Hyperparameters                  | Value           |
| -------------------------------- | --------------- |
| ratio of (train, validate, test) | (0.8, 0.1, 0.1) |
| batch size                       | 32              |
| lr                               | 0.001           |
| n_mfcc                           | 13              |
| n_segments                       | 5               |

#### Results (run on Colab)

| Model                                              | Accuracy     |
| -------------------------------------------------- | ------------ |
| FCN                                                | 50 - 52%     |
| FCN (p=0.3, weight_decay=0.001)                    | 58%          |
| CNN1, CNN2 (initial weights), CNN_Add (add_module) | 59.2 - 60.4% |
| res1                                               | 60.4%        |
| CRNN (p=0.75)                                      | 83.6%        |

p: dropout ratio

weight_decay: regularization factor
