import sys
from deep_learning.src.training import Trainer
from deep_learning.src.evaluation import Evaluator
from deep_learning.src.prediction import Predictor

_predictor = None
def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def run_training():
    t = Trainer()
    train_ds, val_ds = t.load("train"), t.load("validation")
    n = len(train_ds.class_names)
    print("Classes:", train_ds.class_names)
    cnn = t.build_cnn(n)
    t.train(cnn, train_ds, val_ds, epochs=3)
    t.save(cnn, "cnn_model.keras")
    tl = t.build_tl(n)
    t.train(tl, train_ds, val_ds, epochs=3)
    t.save(tl, "tl_model.keras")


def run_eval():
    Evaluator().compare(Trainer().load("test"))



def predict_cnn(file_path):
    label, conf = _get_predictor().predict_cnn(file_path)
    return {"model": "CNN", "prediction": label, "confidence": conf}


def predict_tl(file_path):
    label, conf = _get_predictor().predict_tl(file_path)
    return {"model": "TL", "prediction": label, "confidence": conf}


def predict_both(file_path):
    return _get_predictor().predict_both(file_path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "train":
        run_training()
    elif cmd == "eval":
        run_eval()
    elif cmd == "cnn":
        print(predict_cnn(sys.argv[2]))
    elif cmd == "tl":
        print(predict_tl(sys.argv[2]))
    elif cmd == "predict":
        print(predict_both(sys.argv[2]))
    else:
        run_training()
        run_eval()