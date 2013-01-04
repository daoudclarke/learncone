# Bismillahi-r-Rahmani-r-Rahim
#
# Compute a vector lattice basis

import numpy as np

class Lattice:
    def __init__(self, basis):
        """
        Takes a matrix like object with basis vectors in columns
        """
        self.dimensions = len(basis)
        self.basis_matrix = np.matrix(basis)
        self.basis_inverse = np.linalg.inv(self.basis_matrix)


    def meet(self, u, v):
        """Compute the vector lattice meet of two vectors
        >>> l = Lattice([[1,2],[2,1]])
        >>> l.meet([1,0],[0,1])
        matrix([[-1.],
                [-1.]])
        """
        return self.lattice_op(u, v, np.minimum)

    def join(self, u, v):
        """Compute the vector lattice join of two vectors"""
        return self.lattice_op(u, v, np.maximum)

    def ge(self, u, v):
        """Is u >= v?"""
        u1 = self.basis_inverse*np.matrix(u).T
        v1 = self.basis_inverse*np.matrix(v).T
        return (u1 >= v1).all()

    def lattice_op(self, u, v, op):
        u1 = self.basis_inverse*np.matrix(u).T
        v1 = self.basis_inverse*np.matrix(v).T
        m = op(u1, v1)
        r = self.basis_matrix*m
        return r

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
