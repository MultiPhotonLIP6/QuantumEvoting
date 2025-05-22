import numpy as np

PAULI = {"i": np.array([[1,0],
                        [0,1]]),
        
        "x": np.array([[0,1],
                        [1,0]]),

        "y": np.array([[0,-1j],
                        [1j,0]]),

        "z": np.array([[1,0],
                        [0,-1]]),
        
        "v":np.array([[0,1],
                        [1,0]]),
}
