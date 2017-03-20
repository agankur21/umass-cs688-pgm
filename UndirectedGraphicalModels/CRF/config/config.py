import os
MODEL_PATH_FEATURES=os.path.join(os.getcwd(), "../../model", "feature-params.txt")
MODEL_PATH_TRANSITION=os.path.join(os.getcwd(), "../../model", "transition-params.txt")
NUM_FEATURES=321
DATA_FOLDER=os.path.join(os.getcwd(),"../../data")
DECODING_KEY=list("etainoshrd")
ENCODING_KEY={DECODING_KEY[i]:i for i in range(len(DECODING_KEY))}
