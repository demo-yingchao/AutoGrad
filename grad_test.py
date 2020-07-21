import numpy as np
import grad as g
import pickle

def gaussian():
  u = g.Var(shape=(2,))
  half_sigma = g.Low(shape=(2,2))
  sigma = half_sigma@half_sigma.T
  var = [u, half_sigma]
  x = g.Inp(name='x', shape=(2,))
  neg_log_pdf = 1/2*g.log(g.det(sigma))+1/2*(x-u)@g.inv(sigma)@(x-u).T
  log_like = g.trmean(neg_log_pdf)

  data = np.random.multivariate_normal(
      mean = np.array([3, -3]),
      cov = np.array([[5,-2],[-2,2]]),
      size = (1000,),
      check_valid='warn')

  print('real param')
  print(np.array([3, -3]))
  print(np.array([[5, -2], [-2,2]]))

  inp_dict={'x':data}
  lr = 0.1
  for i in range(600):
    ret = log_like.forward(inp_dict)
    if (i+1)%100 == 0:
      print(ret)
      print(u.val)
      sigma = half_sigma.val@half_sigma.val.T
      print(sigma)
    log_like.update()
    for v in var:
      v.apply_gradient(lr)
  return


if __name__ == '__main__':
  gaussian()
  
  
