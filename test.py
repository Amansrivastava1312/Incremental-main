from ml_tech.scripts.model_save import train_and_save_models
from nlp_sentiment.distil_bert_test import predict

#run this for ML model 
train_and_save_models()



from nlp_sentiment.scripts import (
    log,
    run_step,
    ensure_folders,
    maybe_install,
    main
)

# Call individual functions

log("Testing setup")

ensure_folders()

maybe_install()

# Example custom step
run_step(
    "Train Logistic Regression",
    'python src/train_logreg.py'
)

# Or run everything
main()