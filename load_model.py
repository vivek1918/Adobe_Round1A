import os
import layoutparser as lp

# Optional if you're still using iopath workaround
os.environ["IOPATH_CACHE_DIR"] = os.path.abspath("./iopath_cache")

model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path="./PubLayNet_model/model_final.pth",
    extra_config={"MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.8},
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

print("✅ Model loaded successfully from disk.")
