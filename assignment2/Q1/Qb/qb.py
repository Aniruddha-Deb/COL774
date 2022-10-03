import os
import sys
import numpy as np

if __name__ == "__main__":
    # just need the size of the positive and negative reviews to make these 
    # judgements

    p = len(os.listdir(f"{sys.argv[2]}/pos"))
    n = len(os.listdir(f"{sys.argv[2]}/neg"))

    rp = np.random.binomial(p,0.5)
    rn = np.random.binomial(n,0.5)
    print(f"Random guessing accuracy: {rp+rn}/{p+n} = {(rp+rn)/(p+n)}")
    print(f"Positive guessing accuracy: {p}/{p+n} = {p/(p+n)}")
