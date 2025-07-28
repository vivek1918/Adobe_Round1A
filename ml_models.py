from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import layoutparser as lp
import torch
import os

class PDFMLModels:
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.ml_model = None
        self.layout_model = None
        self.min_confidence = 0.5
        self._initialize_ml_model()
        self._initialize_layout_model()

    def _initialize_ml_model(self):
        """Improved ML model using TF-IDF and RandomForest"""
        if self.use_ml:
            headings = [
                "Introduction", "Chapter 1 Introduction", "1.1 Motivation", "2.1 Methods", "Conclusion",
                "Abstract", "Bibliography", "References", "Figure 1: Overview", "Table 2: Results"
            ]
            non_headings = [
                "This is a sample paragraph explaining the methodology.",
                "In recent years, artificial intelligence has grown rapidly.",
                "The experiment was conducted over 10 days.",
                "We thank our supervisors for their help.",
                "This paper focuses on multiple objectives."
            ]
            X = headings + non_headings
            y = [1] * len(headings) + [0] * len(non_headings)

            self.ml_model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            self.ml_model.fit(X, y)
            
    def _initialize_layout_model(self):
        """Initialize LayoutParser model for document layout analysis"""
        try:
            self.layout_model = lp.Detectron2LayoutModel(
                config_path="/app/PubLayNet_model/config.yml",
                model_path="/app/PubLayNet_model/model_final.pth",
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.min_confidence],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            print(f"Layout model initialization failed: {str(e)}")
            self.layout_model = None