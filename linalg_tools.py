import numpy as np


def default_scal_prod(u,v):
  return np.sum(u*v)


class Base:
  """
  Operations on a vector base. You can use your own objects
  They must implement sum, product with a scalar
  and a scalar product function to be given as argument.

  This will NOT work if the vectors are not orthogonal
  The vectors do not have to be normalized
  """
  def __init__(self,vecs,scal_prod=default_scal_prod,check_ortho=True):
    """
    vecs is a list of ORTHOGONAL vectors
    They don't have to be normalized
    scal_prod is the scalar product defined for your space
    """
    self.vecs = vecs
    self.s = scal_prod
    if check_ortho and len(vecs) > 1:
      self.check_ortho()
    self.n2 = [self.s(vec,vec) for vec in self.vecs]

  def check_ortho(self):
    from itertools import combinations
    norms = [self.s(v,v) for v in self.vecs]
    #s = [abs(self.s(a,b)) for a,b in combinations(self.vecs,2)]
    s = [abs(self.s(va,vb)/(na*nb)) for (va,na),(vb,nb) in
        combinations(zip(self.vecs,norms),2)]
    #print(s)
    if max(s) > 1e-12:
      print("WARNING: Base does not seem orthogonal !")
      print(s,"Maxi=%f"%max(s))
      input("Continue?")

  def get_vec(self,coord):
    """
    Takes coordinates

    Returns the vector with these coordinates in our base
    """
    assert len(coord) == len(self.vecs),"Invalid dimension !"
    return sum(k*v for k,v in zip(coord,self.vecs))

  def proj(self,vec):
    """
    Takes a single vector

    Returns the coordinates of this vector in our base
    """
    return np.array([self.s(vec,i)/n for i,n in zip(self.vecs,self.n2)])

  def proj_vec(self,vec):
    """
    Takes a single vector

    Returns the vector projected in our base
    """
    return self.get_vec(self.proj(vec))


def orthonormalize(vecs,scal=default_scal_prod):
  """
  Not used
  """
  v0 = vecs[0]/scal(vecs[0],vecs[0])**.5
  b = Base([v0])
  for v in vecs[1:]:
    newvec = v-b.proj_vec(v)
    u = scal(newvec,newvec)**.5
    b = Base(b.vecs+[newvec/u],scal)
    assert scal(b.vecs[-1],b.vecs[-1]) != 0,"useless vector: "+str(v)
  return b.vecs


def orthogonalize(vecs,scal=default_scal_prod):
  b = Base(vecs[:1])
  #print("A",b.vecs[-1])
  for v in vecs[1:]:
    b = Base(b.vecs+[v-b.proj_vec(v)],scal,check_ortho=False)
    #print("A",b.vecs[-1])
    assert scal(b.vecs[-1],b.vecs[-1]) != 0,"useless vector: "+str(v)
  return b.vecs


class NonOrthoBase(Base):
  def __init__(self,vecs,scal_prod=default_scal_prod):
    Base.__init__(self,vecs,scal_prod,check_ortho=False)
    self.orthobase = Base(orthogonalize(vecs,scal_prod),scal_prod)
    self.p = np.array([self.orthobase.proj(v) for v in self.vecs])
    self.ip = np.linalg.inv(self.p)

  def proj(self,vec):
    """
    Takes a single vector

    Returns the coordinates of this vector in our base
    """
    return np.dot(self.orthobase.proj(vec),self.ip)


if __name__ == '__main__':
  ux = np.array([1,-1,0,0,0,0])
  uy = np.array([0,0,2,1,0,0])
  uz = np.array([0,0,0,0,1,1])

  #b = Base([ux,uy,uz])
  b = Base([ux,uy])
  print(b.proj_vec(ux+uy-uz))
  print(b.proj(ux+2*uy+3*uz))

  ux2 = ux
  uy2 = uy+2*uz
  uz2 = -uz
  b2 = NonOrthoBase([ux2,uy2,uz2])
  print(b2.proj(ux2+uy2+uz2))
  #print(b2.orthobase.proj(ux+uy+uz))
