import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
import sys
import pickle

def load_data(dir, type):
    data_file = open(f'{dir}/{type}.pickle', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    s = data['data'].shape
    data['data'] = data['data'].reshape(s[0], np.product(s[1:]))

    data['data'] = data['data']
    data['labels'] = data['labels'].flatten()
    
    return data

def save_mismatches(preds, dataset, prefix):
    trues = dataset['labels'].flatten()
    idxs = np.where(preds != trues)[0]
    random_idxs = np.random.choice(idxs, size=10, replace=False)
    vecs = [dataset['data'][i].reshape(32,32,3) for i in random_idxs]
    for (i,vec) in enumerate(vecs):
        img = Image.fromarray(vec)
        img = img.resize((320,320), resample=Image.NEAREST)
        img.save(f"mc_{prefix}_{int(preds[random_idxs[i]])}{int(trues[random_idxs[i]])}_{i+1}.png")

if __name__ == "__main__":
    test_data = load_data(sys.argv[2], 'test_data')

    preds = pickle.load(open('preds.pkl', 'rb'))
    preds_sk = pickle.load(open('preds_sk.pkl', 'rb'))

    # confusion matrices
    sk_cmat = ConfusionMatrixDisplay.from_predictions(preds_sk, test_data['labels'])
    qp_cmat = ConfusionMatrixDisplay.from_predictions(preds, test_data['labels'])
    sk_cmat.figure_.savefig('sk_cmat.pdf', bbox_inches='tight')
    qp_cmat.figure_.savefig('qp_cmat.pdf', bbox_inches='tight')

    save_mismatches(preds, test_data, "qp")
    save_mismatches(preds_sk, test_data, "sk")
